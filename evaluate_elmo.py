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
def restore_collection(path, scopename, sess):
    # retrieve all variables under scope
    variables = {v.name: v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scopename)}
    # retrieves all variables in checkpoint
    for var_name, _ in tf.contrib.framework.list_variables(path):
        # get the value of the variable

        var_value = tf.contrib.framework.load_variable(path, var_name)
        # construct expected variablename under new scope
        target_var_name = '%s/%s:0' % (scopename, var_name)
        # reference to variable-tensor
        if target_var_name not in variables.keys():
            print("skipping {}".format(var_name))
            continue
        target_variable = variables[target_var_name]
        # assign old value from checkpoint to new variable
        print(target_variable, target_var_name)
        sess.run(target_variable.assign(var_value))

class PredictionWrapper():
    def __init__(self, args, queue, session, path=None, graph=None):
        self.args = args
        self.shapes = args.shapes
        self.queue = queue
        self.session = session
        self.outputs = []
        with session.graph.as_default():
            feature_tokens = self.args.shapes.max_buffer_size + self.args.shapes.max_stack_size
            self.form_indices = tf.placeholder(name="form_indices", shape=[None, feature_tokens], dtype=tf.int32)
            self.dep_types = tf.placeholder(name="dep_types", shape=[None, feature_tokens], dtype=tf.int32)
            self.head_indices = tf.placeholder(name="head_indices", shape=[None, feature_tokens], dtype=tf.int32)
            self.pos = tf.placeholder(name="pos", shape=[None, feature_tokens], dtype=tf.int32)
            self.child_indices = tf.placeholder(name="child_indices", shape=[None], dtype=tf.int32)
            self.child_ids = tf.placeholder(name="child_ids", shape=[None], dtype=tf.int32)
            self.child_edge_types = tf.placeholder(name="child_edge_types", shape=[None], dtype=tf.int32)
            self.child_edge_ids = tf.placeholder(name="child_edge_ids", shape=[None], dtype=tf.int32)
            self.batch_ind = tf.placeholder(name="batch_ind", shape=[None], dtype=tf.int32)
            self.ner = tf.placeholder(name="ner", shape=[None, feature_tokens], dtype=tf.int32)
            self.height = tf.placeholder(name="height1", shape=[None, feature_tokens], dtype=tf.int32)
            self.inc = tf.placeholder(name="inc1", shape=[None, feature_tokens, self.args.num_edges], dtype=tf.int32)
            self.out = tf.placeholder(name="out1", shape=[None, feature_tokens, self.args.num_edges], dtype=tf.int32)
            self.history = tf.placeholder(name="hist", shape=[None, None], dtype=tf.int32)
            self.sentence_lengths = tf.placeholder(name="sent_lens", shape=[None], dtype=tf.int32)
            self.action_ratios = tf.placeholder(name="action_ratios", shape=[None], dtype=tf.float32)
            self.node_ratios = tf.placeholder(name="node_ratios", shape=[None], dtype=tf.float32)
            self.action_counts = tf.placeholder(name="act_counts", shape=[None, self.args.num_labels],
                                                dtype=tf.int32)
            self.root = tf.placeholder(name="root1", shape=[None, feature_tokens], dtype=tf.int32)
            self.elmo = tf.placeholder(name="elmo", shape=[None, None, 1024], dtype=tf.float32)
            # [Variable and model creation goes here.]
            for k, path in enumerate(path):
                with tf.variable_scope('model_%03i' % (k + 1)):
                    with tf.variable_scope("model"):
                        self.model = ElModel(args, args.num_labels, args.num_deps, args.num_pos, args.num_ner,train=False,predict=True,batch=(
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
            self.action_counts,
            self.action_ratios,
            self.node_ratios,
            self.root,
            ))
                        self.logits = self.model.logits
                        self.outputs.append(self.logits)
                restore_collection(path,'model_%03i' % (k + 1),session)

            self.ensemble = tf.reduce_mean(self.outputs,axis=0)


        self.num_feature_tokens = self.shapes.max_stack_size + self.shapes.max_buffer_size

    def score(self, features):
        return self.session.run(self.ensemble, feed_dict={
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
            self.elmo:features['elmo'],
            self.node_ratios:features['node_ratios'],
            self.action_ratios:features['action_ratios'],
            self.action_counts:features['action_counts'],
            self.root:features['root']
        })

def dict2namespace(obj):
    if isinstance(obj, dict):
        return Namespace(**{k:dict2namespace(v) for k, v in obj.items()})
    elif isinstance(obj, (list, set, tuple, frozenset)):
        return [dict2namespace(item) for item in obj]
    else:
        return obj
import toml
def evaluate(args):
    model_args = load_args(args.meta_dir)
    # Merge model args with evaluation args
    nt = dict2namespace(toml.load("config.toml"))
    args = Namespace(**{**vars(model_args), **vars(args), **vars(nt)})
    # restore numberers
    with open(os.path.join(args.meta_dir, tuepa.util.config.LABELS_FILENAME), "r", encoding="utf-8") as file:
        label_numberer = load_numberer_from_file(file)

    with open(os.path.join(args.meta_dir, tuepa.util.config.EDGE_FILENAME), "r", encoding="utf-8") as file:
        edge_numberer = load_numberer_from_file(file)

    with open(os.path.join(args.meta_dir, tuepa.util.config.DEP_FILENAME), "r", encoding="utf-8") as file:
        dep_numberer = load_numberer_from_file(file)

    with open(os.path.join(args.meta_dir, tuepa.util.config.POS_FILENAME), "r", encoding="utf-8") as file:
        pos_numberer = load_numberer_from_file(file)

    with open(os.path.join(args.meta_dir, tuepa.util.config.NER_FILENAME), "r", encoding="utf-8") as file:
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
    sess =  tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    paths = tf.train.get_checkpoint_state(os.path.join(args.save_dir, "save_dir")).all_model_checkpoint_paths
    wrp = PredictionWrapper(args=args, queue=None, session=sess,path=paths)

    try:
        if args.test:
            tf.logging.info("Start to parse test passages!")
            parser.parse(wrp,args,read_passages(args.eval_data))
        else:
            res = list(parser.evaluate(wrp, args,read_passages(args.eval_data)))
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
