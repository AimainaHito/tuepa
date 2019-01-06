import os

import parser
import tensorflow as tf

from ucca import constructions

from argparse import Namespace
import config
from config import create_argument_parser, save_args, load_args
from model import ElModel
from numberer import Numberer, load_numberer_from_file
from preprocessing import read_passages
from collections import namedtuple

ElmoPredictionData = namedtuple("ElmoPredictionData","label_numberer pos_numberer dep_numberer edge_numberer")

class PredictionWrapper:
    def __init__(self, estimator,shapes,args):
        self.estimator = estimator
        self.shapes = shapes

    def score(self, features):
        return self.estimator.predict(tf.estimator.inputs.numpy_input_fn(features,shuffle=False))


def evaluate(args):
    model_args = load_args(args.model_dir)
    # Merge model args with evaluation args
    args = Namespace(**{**vars(args), **vars(model_args)})

    # restore numberers
    with open(os.path.join(args.save_dir, config.LABELS_FILENAME), "r", encoding="utf-8") as file:
        label_numberer = load_numberer_from_file(file)

    with open(os.path.join(args.save_dir, config.EDGE_FILENAME), "r", encoding="utf-8") as file:
        edge_numberer = load_numberer_from_file(file)

    with open(os.path.join(args.save_dir, config.DEP_FILENAME), "r", encoding="utf-8") as file:
        dep_numberer = load_numberer_from_file(file)

    with open(os.path.join(args.save_dir, config.POS_FILENAME), "r", encoding="utf-8") as file:
        pos_numberer = load_numberer_from_file(file)

    args.num_edges = edge_numberer.max

    def eval_model_fn(features, labels, mode, params):
        args = params["args"]

        model = ElModel(args, label_numberer.max, dep_numberer.max, pos_numberer.max)
        logits = model(features, train=mode == tf.estimator.ModeKeys.TRAIN, mode=mode)
        predictions = tf.argmax(logits, -1)
        result = {
            "classes": predictions,
            "probabilities": tf.nn.softmax(logits),
        }

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=logits,
            export_outputs={
                "classify": tf.estimator.export.PredictOutput(result)
            })

    run_conf = tf.estimator.RunConfig(model_dir=args.model_dir,
                                      save_summary_steps=100,
                                      save_checkpoints_steps=10000,
                                      log_step_count_steps=20,
                                      train_distribute=None)
    estimator = tf.estimator.Estimator(
        model_fn=eval_model_fn,
        model_dir=args.save_dir,
        config=run_conf,
        params={"args": args})

    args.prediction_data = ElmoPredictionData(
        label_numberer=label_numberer,
        pos_numberer=pos_numberer,
        dep_numberer=dep_numberer,
        edge_numberer=edge_numberer
    )

    wrp = PredictionWrapper(estimator,args.shapes,args)
    print(*parser.evaluate(wrp, args, read_passages([args.eval_data])), sep="\n")


def run_eval(args):
    argparser = config.get_oracle_parser()
    constructions.add_argument(argparser)
    config.get_eval_parser(argparser)
    args = argparser.parse_args(args)
    args.model_type="elmo-rnn"
    evaluate(args)


if __name__ == "__main__":
    import sys
    run_eval(sys.argv[1:])
