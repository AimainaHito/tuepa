import multiprocessing
import os

import parser
import tensorflow as tf

from ucca import constructions

from argparse import Namespace

import tuepa.util.config

from tuepa.util.config import create_argument_parser, save_args, load_args, get_oracle_parser, get_eval_parser
from tuepa.model import ElModel
from tuepa.util.numberer import Numberer, load_numberer_from_file
from tuepa.data import read_passages
from collections import namedtuple

ElmoPredictionData = namedtuple(
    "ElmoPredictionData", "label_numberer pos_numberer dep_numberer edge_numberer")


class PredictionWrapper():
    def __init__(self, args,queue, session):
        self.shapes = args.shapes
        self.queue = queue
        self.model = ElModel(args, args.num_labels,
                        args.num_deps, args.num_pos)
        features = self.inputs()
        self.logits = self.model(features, train=False)
        self.predictions = tf.argmax(self.logits, -1)
        self.session = session
        self.loaded = False

    def inputs(self):
        feature_tokens = self.shapes.max_stack_size + self.shapes.max_buffer_size
        self.form_indices = tf.placeholder(name="form_indices", shape=[None, feature_tokens],dtype=tf.int32)
        self.dep_types = tf.placeholder(name="dep_types", shape=[None, feature_tokens],dtype=tf.int32)
        self.head_indices = tf.placeholder(name="head_indices", shape=[None, feature_tokens],dtype=tf.int32)
        self.pos = tf.placeholder(name="pos", shape=[None, feature_tokens],dtype=tf.int32)
        self.height = tf.placeholder(name="height", shape=[None, feature_tokens],dtype=tf.int32)
        self.inc = tf.placeholder(name="inc", shape=[None, feature_tokens, self.args.num_edges],dtype=tf.int32)
        self.out = tf.placeholder(name="out", shape=[None, feature_tokens, self.args.num_edges],dtype=tf.int32)
        self.history = tf.placeholder(name="hist", shape=[None, None],dtype=tf.int32)
        self.elmo = tf.placeholder(name="elmo", shape=[None, None, 1024],dtype=tf.float32)
        self.sentence_lengths = tf.placeholder(name="sent_lens", shape=[None],dtype=tf.float32)
        self.history_lengths = tf.placeholder(name="hist_lens", shape=[None],dtype=tf.float32)
        self.action_counts = tf.placeholder(name="act_counts", shape=[None,self.args.num_labels], dtype=tf.int32)
        return self.dep_types, self.elmo, self.form_indices, self.head_indices, self.height, self.history, self.history_lengths, self.inc, self.out, self.pos, self.sentence_lengths, self.action_counts
    
    def score(self, features):
        if not self.loaded:
            tf.train.Saver.restore(self.session)
            self.loaded = True
        return self.session.run(self.logits, feed_dict={ self.form_indices:features['form_indices'],
            self.dep_types:features['deps'],
            self.pos:features['pos'],
            self.head_indices:features['heads'],
            self.height:features['height'],
            self.inc:features['inc'],
            self.out:features['out'],
            self.history:features['history'],
            self.sentence_lengths:features['sent_lens'],
            self.history_lengths:features['hist_lens'],
            self.elmo:features['elmo'],
            self.action_counts:features['action_counts']
        })

def evaluate(args):
    model_args = load_args(args.model_dir)
    # Merge model args with evaluation args
    args = Namespace(**{**vars(args), **vars(model_args)})

    # restore numberers
    with open(os.path.join(args.save_dir, tuepa.config.LABELS_FILENAME), "r", encoding="utf-8") as file:
        label_numberer = load_numberer_from_file(file)

    with open(os.path.join(args.save_dir, tuepa.config.EDGE_FILENAME), "r", encoding="utf-8") as file:
        edge_numberer = load_numberer_from_file(file)

    with open(os.path.join(args.save_dir, tuepa.config.DEP_FILENAME), "r", encoding="utf-8") as file:
        dep_numberer = load_numberer_from_file(file)

    with open(os.path.join(args.save_dir, tuepa.config.POS_FILENAME), "r", encoding="utf-8") as file:
        pos_numberer = load_numberer_from_file(file)

    args.num_edges = edge_numberer.max
    args.prediction_data = ElmoPredictionData(
        label_numberer=label_numberer,
        pos_numberer=pos_numberer,
        dep_numberer=dep_numberer,
        edge_numberer=edge_numberer
    )
    with tf.Session() as sess:
        wrp = PredictionWrapper(args=args,queue=None,session=sess)
        print(*parser.evaluate(wrp, args,
                            read_passages([args.eval_data])), sep="\n")


def run_eval(args):
    argparser = get_oracle_parser()
    constructions.add_argument(argparser)
    get_eval_parser(argparser)
    args = argparser.parse_args(args)
    args.model_type = "elmo-rnn"
    evaluate(args)


if __name__ == "__main__":
    import sys
    run_eval(sys.argv[1:])
