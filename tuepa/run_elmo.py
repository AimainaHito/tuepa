import copy
import json
import multiprocessing
import os
import pickle

import tensorflow as tf
import numpy as np
import h5py
import finalfrontier

from timeit import default_timer

from config import create_argument_parser, save_args, load_args, ARGS_FILENAME
from elmoformanylangs import Embedder
from model import ElModel, feed_forward_from_json
from numberer import Numberer, load_numberer_from_file
from preprocessing import preprocess_dataset
from util import SaverHook

LABELS_FILENAME = "labels.csv"

def get_estimator(args, label_numberer):
    """
    Returns an estimator object.
    :param args: named_tuple holding commandline arguments.
    :param label_numberer:
    :return: tuple: (Estimator, tf.estimator.TrainSpec, tf.estimator.EvalSpec)
    """

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
        ff_layers = params["ff_layers"]
        num_labels = params["num_labels"]
        with h5py.File(args.validation_path, 'r') as data:
            model = ElModel(args, ff_layers, num_labels, json.loads(data['shapes'].value))

        if mode == tf.estimator.ModeKeys.TRAIN:
            metrics, grads_vars = model.compute_gradients(features, labels)
            train_op = model.optimizer.apply_gradients(grads_vars, global_step=tf.train.get_or_create_global_step())

            accuracy, x_ent = metrics

            return tf.estimator.EstimatorSpec(mode=mode, loss=tf.reduce_mean(x_ent), train_op=train_op)
        else:

            logits = model(features, train=mode == tf.estimator.ModeKeys.TRAIN)
            predictions = tf.argmax(logits, -1)

            if mode == tf.estimator.ModeKeys.EVAL:
                hooks = [SaverHook(
                    labels=label_numberer.num2value,
                    confusion_matrix_tensor_name='mean_iou/total_confusion_matrix',
                    summary_writer=tf.summary.FileWriterCache.get(str("sv_dir/eval_validation"))
                )]

                on_h_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.int32)
                on_h_preds = tf.one_hot(predictions, depth=num_labels)

                evalMetrics = {'accuracy': tf.metrics.accuracy(labels, predictions),
                               'precision': tf.metrics.precision(on_h_labels, on_h_preds),
                               'f1': tf.contrib.metrics.f1_score(on_h_labels, on_h_preds),
                               'recall': tf.metrics.recall(on_h_labels, on_h_preds),
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

    def get_elmo_input_fn(data_path, data_shapes, train_or_eval, args, train):
        """
        Returns the input_fn for the elmo model. Starts a background process that reads and preprocesses chunks from
        HDF5 files.

        Feature ordering: (stack_buffer, history, elmo, padding, nonterminal,sentence_lens, history_lens)

        :param data_shapes: tuple describing containing shape information of input tensors
        :param batch_size: mini batch size.
        :param train_or_eval: train and eval input fn are identical while predict differs.
        :param embedding_size: size of finalfrontier embeddings
        :param embedder: path to finalfrontier embeddings.
        :param train: indicator if training is running. Training means shuffled input.
        :return: a callable which returns a tf.Dataset build from a generator.
        """
        q = multiprocessing.Queue(maxsize=4)
        p = multiprocessing.Process(target=h5py_worker, args=(data_path, q, args))
        p.daemon = True
        p.start()

        def get_dataset():
            def generator():
                # listen forever
                while True:
                    for item in zip(*q.get()):
                        # ((features), labels)
                        yield tuple((item[:-1], item[-1]))

            d = tf.data.Dataset.from_generator(generator, output_types=
            # ((stack_buffer, history, elmo, padding, nonterminal,sentence_lens, history_lens),labels)
            ((tf.float32, tf.int32, tf.float32, tf.bool, tf.bool, tf.int32, tf.int32), tf.int32))
            if train:
                return d.shuffle(args.batch_size * 10)
            else:
                return d.repeat()

        if train_or_eval:
            return lambda: get_dataset().padded_batch(args.batch_size, data_shapes, drop_remainder=True).prefetch(1)
        else:
            pass

    run_conf = tf.estimator.RunConfig(model_dir=args.save_dir,
                                      save_summary_steps=20,
                                      save_checkpoints_steps=args.epoch_steps,
                                      log_step_count_steps=20,
                                      train_distribute=None)

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=args.save_dir,
        config=run_conf,
        params={"num_labels": args.num_labels, "ff_layers": args.layers, "args": args})

    with h5py.File(args.validation_path, 'r') as data:
        data_shapes = (
            (data['stack_and_buffer_features'][0].shape + (args.embedding_size,),   # stack and buffer
             tf.TensorShape([None]),                                                # history
             tf.TensorShape([None, 1024]),                                          # elmo
             tf.TensorShape(
                 data['stack_and_buffer_features'][0].shape),                       # padding mask
             tf.TensorShape(
                 data['stack_and_buffer_features'][0].shape),                       # non terminals
             tf.TensorShape([]),                                                    # sentence lengths
             tf.TensorShape([])),                                                   # history lengths
            tf.TensorShape([]))                                                     # labels

    train_spec = tf.estimator.TrainSpec(
        get_elmo_input_fn(args.training_path, data_shapes, True, args=args, train=True),
    )
    eval_spec = tf.estimator.EvalSpec(
        get_elmo_input_fn(args.validation_path, data_shapes, True, args=args, train=False),
        steps=args.epoch_steps,
        throttle_secs=60,
        name='validation',
    )
    return estimator, train_spec, eval_spec
#
# class PredictionWrapper:
#     def __init__(self,estimator):
#         self.estimator = estimator
#
#     def score(self,features):
#         self.estimator.
#         feature_spec = {'foo': tf.FixedLenFeature(...),
#                         'bar': tf.VarLenFeature(...)}
#
#         def serving_input_receiver_fn():
#             """An input receiver that expects a serialized tf.Example."""
#             serialized_tf_example = tf.placeholder(dtype=tf.string,
#                                                    shape=[None],
#                                                    name='input_example_tensor')
#             receiver_tensors = {'examples': serialized_tf_example}
#             features = tf.parse_example(serialized_tf_example, feature_spec)
#             return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)


def h5py_worker(data_path, queue, args):
    """
    Method to read chunks from HDF5 files containing examples.

    Opens a HDF5 file at `data_path`, reads and preprocesses 10 batches at once and sends puts them into `queue`.

    :param data_path: path to HDF5 file.
    :param queue: a queue object
    :param args: named tuple holding commandline arguments and other information
    """

    embedder = finalfrontier.Model(args.ff_path, mmap=True)

    def prepare(data, index=None):
        state2pid = np.array(data['state2sent_index'])
        getters = list(range(index * args.batch_size*10, min((index + 1) * args.batch_size*10, len(state2pid))))

        ids = state2pid[getters]
        elmos = dict()
        batch_elmo = []
        for i in ids:
            if i in elmos:
                batch_elmo.append(elmos[i])
            else:
                res = data['elmo'][str(i).encode("UTF-8")].value
                elmos[i] = res
                batch_elmo.append(elmos[i])

        batch_elmo = np.array(batch_elmo)
        max_length = max(map(len, batch_elmo))
        # loop over sentences and pad them to max length in batch
        batch_elmo = [np.vstack([n, np.zeros(
            shape=[max_length - len(n),
                   n.shape[-1]],
            dtype=np.float32)])
                      for n in batch_elmo]

        # [batch, time, D]
        # batch_elmo = np.transpose(batch_elmo, [0, 1, 2])
        # Map words on stack and buffer to their embeddings. Mapping them ahead of time means excessive use of memory.
        # TODO: find smart way to batch embeddings ahead of time or use threading to prepare batches during the tf calls
        batch_features = data['stack_and_buffer_features'][getters]
        embedding_collector = []
        for f in batch_features:
            vec = []
            for e in f:
                if e == '':
                    vec.append(args.embedding_size * [0.])
                elif e == "<NT>":
                    vec.append(args.embedding_size * [1.])
                else:
                    vec.append(embedder.embedding(e))
            embedding_collector.append(vec)

        batch_features = np.array(embedding_collector, dtype=np.float32)
        padding = np.isclose(batch_features.mean(axis=-1), 0.)
        non_terminals = np.isclose(batch_features.mean(axis=-1), 1.)
        return batch_features, data['history_features'][getters], batch_elmo, padding, non_terminals, \
               data['sentence_lengths'].value[ids], data['history_lengths'][getters], data['labels'][getters]

    with h5py.File(data_path, 'r') as data:
        max_ind = len(data['labels'])
        index = 0
        next_b = prepare(data, index=index)
        while True:
            if next_b:
                queue.put(next_b)
                next_b = prepare(data, index=index)
                index += 1
                if index == max_ind - args.batch_size*10:
                    index = 0


def train(args):
    """
    Runs the training of the elmo-rnn model.
    :param args: namedtuple holding commandline arguments
    """
    with open(os.path.join(args.save_dir, LABELS_FILENAME), "r", encoding="utf-8") as file:
        label_numberer = load_numberer_from_file(file)
    args.num_labels = label_numberer.max

    if "json_layers" in args:
        args.layers = feed_forward_from_json(json.loads(args.json_layers))

    estimator, train_spec, eval_spec = get_estimator(args, label_numberer=label_numberer)

    tf.logging.info("Training:")
    tf.estimator.train_and_evaluate(
        estimator,
        train_spec,
        eval_spec
    )


def preprocess(args):
    if args.verbose:
        # Print to stderr so it doesn't get piped
        print("Processing passages...", end="\r", file=sys.stderr)
    processing_start_time = default_timer()
    elmo_embedder = Embedder(args.elmo_path, batch_size=32)

    label_numberer = Numberer()
    training_data = preprocess_dataset(
        args.training_path,
        args,
        embedder=elmo_embedder,
        max_features=args.max_training_features,
        label_numberer=label_numberer,
        passage_seperator=None,
        use_elmo=True
    )
    # Preprocess validation set
    validation_data = preprocess_dataset(
        args.validation_path,
        args,
        maximum_feature_size=training_data.shapes,
        embedder=elmo_embedder,
        max_features=args.max_validation_features,
        label_numberer=label_numberer,
        passage_seperator=None,
        use_elmo=True
    )

    # Clear the line before printing over it
    if args.verbose:
        print(end="\033[K", file=sys.stderr)
        print("Processing {} passages complete in {:.4f}s".format(
            len(training_data.sentence_lengths) + len(validation_data.sentence_lengths),
            default_timer() - processing_start_time
        ))

        # Save arguments and dictionaries
    with open(os.path.join(args.save_dir, LABELS_FILENAME), "w", encoding="utf-8") as file:
        label_numberer.to_file(file)
    args.label_list = label_numberer.num2value
    save_args(args, args.save_dir)


def main(args):
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.enable_eager_execution()
    argument_parser = create_argument_parser()

    args = argument_parser.parse_args(args)

    if args.model_type == "preprocess":
        preprocess(args)
        return
    if args.model_type != "elmo-rnn":
        print("run tuepa.py to train, preprocess or evaluate!")

    if args.log_file:
        with args.log_file:
            # Log commandline arguments
            args.log_file.write("{}\n".format(args))
            args.layers = feed_forward_from_json(json.loads(args.layers))
            train(args)

    else:
        args.json_layers = args.layers
        args.layers = feed_forward_from_json(json.loads(args.layers))
        print(args.model_type)
        train(args)


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
