import os


import tensorflow as tf

from ucca import constructions

from argparse import Namespace

import tuepa.util.config

from tuepa.util.config import create_argument_parser, save_args, load_args, get_oracle_parser, get_eval_parser
from tuepa.nn import ElModel
from tuepa.util.numberer import Numberer, load_numberer_from_file
from tuepa.data import read_passages
from tuepa.parser import parser
from collections import namedtuple

ElmoPredictionData = namedtuple(
    "ElmoPredictionData", "label_numberer pos_numberer dep_numberer edge_numberer ner_numberer")


class PredictionWrapper():
    def __init__(self, args, queue, session):
        self.args = args
        self.shapes = args.shapes
        self.queue = queue
        self.session = session

        with self.session.graph.as_default():
            # [Variable and model creation goes here.]
            with tf.variable_scope("model"):
                self.model = ElModel(args, args.num_labels, args.num_deps, args.num_pos, args.num_ner,train=False,predict=True)
                self.logits = self.model.logits
                self.saver = tf.train.Saver()
                self.predictions = tf.argmax(self.logits, -1)
                self.saver.restore(self.session, tf.train.latest_checkpoint(os.path.join(self.args.model_dir,"save_dir")))
        (
            self.form_indices,
            self.dep_types,
            self.head_indices,
            self.pos,
            self.child_indices,
            self.child_ids,
            self.child_edge_types,
            self.child_edge_ids,
            self.batch_ind,
            self.ner,
            self.height,
            self.inc,
            self.out,
            self.history,
            self.elmo,
            self.sentence_lengths,
            self.history_lengths,
            self.action_counts,
            self.action_ratios,
            self.node_ratios,
            self.root,
            ) = self.model.inpts
        self.num_feature_tokens = self.shapes.max_stack_size + self.shapes.max_buffer_size

    def score(self, features):
        return self.session.run(self.logits, feed_dict={
            self.form_indices:features['form_indices'],
            self.dep_types:features['deps'],
            self.pos:features['pos'],
            self.child_indices:features['child_indices'],
            self.child_ids:features['child_ids'],
            self.batch_ind:features['child_batch_ids'],
            self.child_edge_ids:features['child_edge_ids'],
            self.child_edge_types:features['child_edge_types'],
            self.ner:features['ner'],
            self.head_indices:features['heads'],
            self.height:features['height'],
            self.inc:features['inc'],
            self.out:features['out'],
            self.history:features['history'],
            self.sentence_lengths:features['sent_lens'],
            self.history_lengths:features['hist_lens'],
            self.elmo:features['elmo'],
            self.node_ratios:features['node_ratios'],
            self.action_ratios:features['action_ratios'],
            self.action_counts:features['action_counts'],
            self.root:features['root']
        })


def evaluate(args):
    print(args.model_dir)
    model_args = load_args(args.model_dir)
    # Merge model args with evaluation args
    print(model_args)
    print(args)
    args = Namespace(**{**vars(model_args), **vars(args)})
    
    # restore numberers
    with open(os.path.join(args.model_dir, tuepa.util.config.LABELS_FILENAME), "r", encoding="utf-8") as file:
        label_numberer = load_numberer_from_file(file)

    with open(os.path.join(args.model_dir, tuepa.util.config.EDGE_FILENAME), "r", encoding="utf-8") as file:
        edge_numberer = load_numberer_from_file(file)

    with open(os.path.join(args.model_dir, tuepa.util.config.DEP_FILENAME), "r", encoding="utf-8") as file:
        dep_numberer = load_numberer_from_file(file)

    with open(os.path.join(args.model_dir, tuepa.util.config.POS_FILENAME), "r", encoding="utf-8") as file:
        pos_numberer = load_numberer_from_file(file)

    with open(os.path.join(args.model_dir, tuepa.util.config.NER_FILENAME), "r", encoding="utf-8") as file:
        ner_numberer = load_numberer_from_file(file)

    args.num_edges = edge_numberer.max
    args.prediction_data = ElmoPredictionData(
        label_numberer=label_numberer,
        pos_numberer=pos_numberer,
        dep_numberer=dep_numberer,
        edge_numberer=edge_numberer,
        ner_numberer=ner_numberer,
    )
    args.num_ner = ner_numberer.max

    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        wrp = PredictionWrapper(args=args, queue=None, session=sess)
        try:
            if args.test:
                tf.logging.info("Start to parse test passages!")
                parser.parse(wrp,args,read_passages([args.eval_data]))
            else:
                res = list(parser.evaluate(wrp, args,read_passages([args.eval_data])))
        except Exception as e:
            raise


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
