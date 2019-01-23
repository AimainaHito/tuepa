import os

import tensorflow as tf
from argparse import Namespace

from tuepa.util.config import create_argument_parser, save_args, load_args, ARGS_FILENAME, LABELS_FILENAME, \
    DEP_FILENAME, EDGE_FILENAME, POS_FILENAME, NER_FILENAME
from tuepa.data.elmo import get_elmo_input_fn
from tuepa.nn import ElModel
from tuepa.util.numberer import Numberer, load_numberer_from_file
from tuepa.util import SaverHook,PerClassHook,EvalHook
from tuepa.data.preprocessing import read_passages
from elmoformanylangs import Embedder
import torch
import numpy as np


def get_estimator(args, label_numberer, edge_numberer, dep_numberer, pos_numberer,ner_numberer):
    """
    Returns an estimator object.
    :param args: named_tuple holding commandline arguments.
    :param label_numberer:
    :return: tuple: (Estimator, tf.estimator.TrainSpec, tf.estimator.EvalSpec)
    """
#    files = list(map(lambda x: "data/dev/UCCA_English-Wiki_XML/" + x,
#                     os.listdir("data/dev/UCCA_English-Wiki_XML")))
#    eval_passages = list(read_passages(files))
#    elmo = Embedder(args.elmo_path,batch_size=1)
#    for p in eval_passages:
#        s = elmo.sents2elmo([[str(n) for n in p.layer("0").all]])[0]
#        p.elmo = [s]
#
#    torch.cuda.empty_cache()
#    del elmo

    def model_fn(features, labels, mode, params):
        """
        Construct the the model.
        :param features:
        :param labels:
        :param mode:
        :param params:
        :return:
        """
        args = params["args"]
        num_labels = params["num_labels"]
        num_deps = params["num_deps"]
        num_pos = params["num_pos"]

        model = ElModel(args, num_labels, num_dependencies=num_deps, num_pos=num_pos,num_ner=ner_numberer.max)

        if mode == tf.estimator.ModeKeys.TRAIN:
            metrics, grads_vars = model.compute_gradients(features, labels)
            train_op = model.optimizer.apply_gradients(grads_vars, global_step=tf.train.get_or_create_global_step())

            accuracy, x_ent = metrics

            return tf.estimator.EstimatorSpec(mode=mode, loss=tf.reduce_mean(x_ent), train_op=train_op)
        else:
            logits = model(features, train=mode == tf.estimator.ModeKeys.TRAIN, mode=mode)
            predictions = tf.argmax(logits, -1)

            if mode == tf.estimator.ModeKeys.EVAL:
                hooks = [SaverHook(
                    labels=label_numberer.num2value,
                    confusion_matrix_tensor_name='mean_iou/total_confusion_matrix',
                    summary_writer=tf.summary.FileWriterCache.get(os.path.join(args.save_dir,"eval_validation"))),
                    PerClassHook(
                        labels=label_numberer.num2value,
                        tensor_name='mean_accuracy/div_no_nan',
                        summary_writer=tf.summary.FileWriterCache.get(os.path.join(args.save_dir, "eval_validation"))),
#                    EvalHook(
#                        args,
#                        model(None, train=mode == tf.estimator.ModeKeys.TRAIN, mode=mode,eval=True),
#                        model.inpts,
#                        eval_passages,
#                        summary_writer=tf.summary.FileWriterCache.get(os.path.join(args.save_dir,"eval_validation"))
#                    )
                    ]

                evalMetrics = {'accuracy': tf.metrics.accuracy(labels, predictions),
                               'mean_per_class_accuracy': tf.metrics.mean_per_class_accuracy(labels,predictions,num_labels),
                               'mean_iou': tf.metrics.mean_iou(labels, predictions, num_labels)}

                loss = tf.reduce_mean(model.loss(labels=labels, logits=logits))

                return tf.estimator.EstimatorSpec(
                    mode=mode,
                    loss=loss,
                    eval_metric_ops=evalMetrics,
                    evaluation_hooks=hooks)
            else:  # Prediction
                result = {
                    "classes": predictions,
                    "probabilities": tf.nn.softmax(logits),
                }

                return tf.estimator.EstimatorSpec(
                    mode=mode,
                    predictions=predictions,
                    export_outputs={
                        "classify": tf.estimator.export.PredictOutput(result)
                    })
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options)
    run_conf = tf.estimator.RunConfig(model_dir=args.save_dir,
                                      save_summary_steps=20,
                                      save_checkpoints_steps=args.epoch_steps,
                                      log_step_count_steps=20,
                                      session_config = config,
                                      train_distribute=None)

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=args.save_dir,
        config=run_conf,
        params={"num_labels": args.num_labels,
                "num_deps": dep_numberer.max,
                "num_pos": pos_numberer.max,
                "ff_layers": args.layers, "args": args})

    train_spec = tf.estimator.TrainSpec(
        get_elmo_input_fn(args.training_path, True, args=args, train=True),
    )
    eval_spec = tf.estimator.EvalSpec(
        get_elmo_input_fn(args.validation_path, True, args=args, train=False),
        steps=args.epoch_steps//2,
        throttle_secs=60,
        name='validation',
    )
    return estimator, train_spec, eval_spec


def train(args):
    """
    Runs the training of the elmo-rnn model.
    :param args: namedtuple holding commandline arguments
    """
    model_args = load_args(args.save_dir)
    # Merge preprocess args with train args
    args = Namespace(**{**vars(model_args),**vars(args)})

    with open(os.path.join(args.save_dir, LABELS_FILENAME), "r", encoding="utf-8") as file:
        label_numberer = load_numberer_from_file(file)

    with open(os.path.join(args.save_dir, EDGE_FILENAME), "r", encoding="utf-8") as file:
        edge_numberer = load_numberer_from_file(file)

    with open(os.path.join(args.save_dir, DEP_FILENAME), "r", encoding="utf-8") as file:
        dep_numberer = load_numberer_from_file(file)

    with open(os.path.join(args.save_dir, POS_FILENAME), "r", encoding="utf-8") as file:
        pos_numberer = load_numberer_from_file(file)

    with open(os.path.join(args.save_dir, NER_FILENAME), "r", encoding="utf-8") as file:
        ner_numberer = load_numberer_from_file(file)

    # save args for eval call
    save_args(args, args.save_dir)

    estimator, train_spec, eval_spec = get_estimator(args,
                                                     label_numberer=label_numberer,
                                                     edge_numberer=edge_numberer,
                                                     dep_numberer=dep_numberer,
                                                     pos_numberer=pos_numberer,
                                                     ner_numberer=ner_numberer)

    tf.logging.info("Training:")
    tf.estimator.train_and_evaluate(
        estimator,
        train_spec,
        eval_spec
    )


def main(args):
    import tuepa.util.config as config
    tf.logging.set_verbosity(tf.logging.INFO)
    argument_parser = config.get_elmo_parser()
    args = argument_parser.parse_args(args)
    train(args)


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
