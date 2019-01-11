import tensorflow as tf
import opennmt as onmt
import numpy as np
from tuepa.util.numberer import Numberer
from tuepa.data.preprocessing import read_passages
from tuepa.util.config import get_preprocess_parser, get_oracle_parser
from ucca import constructions
import sys

# import finalfrontier
# finalfrontier.Model()
def preprocess(passages, label_numberer, char_numberer, train):
    sents = [[node.text for node in p.layer("0").all] for p in passages]

    tokenized = list(map(lambda x: [char_numberer.number(c,train=train) for c in " ".join(x)], sents))

    sents = []
    targets = []
    go = label_numberer.number("<GO>", train=True)
    stop = label_numberer.number("<STOP>", train=True)

    for p, sent in zip(passages, tokenized):

        t = (go,)+label_numberer.number_sequence(str(p).split(), train=train)+(stop,)
        if len(t) > 500:
            continue
        targets.append(t)
    return tokenized, targets

def py2numpy(sents, targets):
    np_sents = np.zeros(
        shape=(len(sents), max(map(len, sents))), dtype=np.int32)
    np_targets = np.zeros(
        shape=(len(sents), max(map(len, targets))), dtype=np.int32)
    np_lens = np.zeros(shape=(len(sents),), dtype=np.int32)
    in_length = np.zeros_like(np_lens)
    n = 0
    for s, t in zip(sents, targets):
        np_sents[n, :len(s)] = s
        np_targets[n, :len(t)] = t
        in_length[n] = len(s)
        np_lens[n] = len(t)
        n += 1
    return np_sents, np_targets, np_lens, in_length

from opennmt.utils.losses import cross_entropy_sequence_loss


class Model:
    def __init__(self, label_numberer, char_numberer, train):
        self.inputs = tf.placeholder(
            name="inputs", shape=[None, None], dtype=tf.int32)
        self.sequence_length = tf.placeholder(
            name="target_lens", shape=[None], dtype=tf.int32)

        self.targets = tf.placeholder(name="targets", shape=[
            None, None], dtype=tf.int32)
        self.target_lens = tf.placeholder(
            name="target_lens", shape=[None], dtype=tf.int32)

        output_vocabulary = tf.get_variable(
            "out_voc", shape=[label_numberer.max, 256], dtype=tf.float32)
        input_vocabulary = tf.get_variable(
            "in_voc", shape=[char_numberer.max, 256], dtype=tf.float32)
        #
        input_embeddings = tf.nn.embedding_lookup(ids=self.inputs,params=input_vocabulary)
        #
        # input_embeddings = tf.layers.dense(self.inputs,256,use_bias=False)

        with tf.variable_scope("encoder"):
            encoder = onmt.encoders.SelfAttentionEncoder(num_layers=4,num_units=256,ffn_inner_dim=1024,num_heads=4)
            outputs, state, outputs_length = encoder.encode(
                input_embeddings,
                sequence_length=self.sequence_length,
                mode=tf.estimator.ModeKeys.TRAIN)
        with tf.variable_scope("decoder"):
            decoder = onmt.decoders.SelfAttentionDecoder(num_layers=4,num_units=256,ffn_inner_dim=1024,num_heads=4)
            target_embeddings = tf.nn.embedding_lookup(output_vocabulary, self.targets)
            if train:
                self.logits, _, _, attention = decoder.decode(
                    target_embeddings[:,:-1],
                    self.target_lens-1,
                    vocab_size=label_numberer.max,
                    initial_state=state,
                    memory=outputs,
                    memory_sequence_length=outputs_length,
                    return_alignment_history=True)
                self.loss = self._compute_loss()
                self.train_op = tf.train.AdamOptimizer(0.001).minimize(self.loss)
            else:
                self.sampled_ids, _, sampled_length, log_probs, alignment = decoder.dynamic_decode(
                    output_vocabulary,
                    tf.tile([1],[tf.shape(outputs)[0]]),
                    2,
                    vocab_size=label_numberer.max,
                    initial_state=state,
                    maximum_iterations=500,
                    minimum_length=10,
                    mode=tf.estimator.ModeKeys.PREDICT,
                    memory=outputs,
                    memory_sequence_length=outputs_length,
                    return_alignment_history=True,)

    def _compute_loss(self):
        loss, loss_normalizer, loss_token_normalizer = cross_entropy_sequence_loss(self.logits,self.targets[:,1:],self.target_lens)
        return loss

if __name__ == "__main__":
    oracle_parser = get_oracle_parser()
    constructions.add_argument(oracle_parser)
    argument_parser = get_preprocess_parser(parents=[oracle_parser])
    argument_parser.add_argument("bert", help="path to bert model.")
    argument_parser.add_argument("-b", "--batch_size", default=16, help="batch size.")

    args = argument_parser.parse_args(sys.argv[1:])

    label_numberer = Numberer(first_elements=["<PAD>","<GO>","<STOP>"])
    char_numberer = Numberer(first_elements=["<PAD>"])
    passages = list(read_passages([sys.argv[1]]))

    train_sents, train_targets,= preprocess(passages=list(read_passages([args.training_path])),label_numberer=label_numberer,char_numberer=char_numberer, train=True)
    val_sents, val_targets = preprocess(passages=list(read_passages([args.validation_path])),label_numberer=label_numberer,char_numberer=char_numberer,train=False)
    print(len(train_sents))
    print(len(val_sents))
    np_sents, np_targets, train_target_lens, train_in_lens = py2numpy(train_sents, train_targets)
    val_np_sents, val_np_targets, val_np_lens, val_np_in_lens= py2numpy(val_sents, val_targets)
    print(len(np_sents))
    print(len(val_np_sents))
    conf = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    best_accuracy = -1
    patience = 0
    conf.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    with tf.Session(config=conf) as sess:

        with tf.name_scope("train"):
            with tf.variable_scope('model', reuse=False):
                train_model = Model(label_numberer,char_numberer,train=True)
        print("Train model done.")
        with tf.name_scope("val"):
            with tf.variable_scope("model", reuse=True):
                val_model = Model(label_numberer, char_numberer,train=False)
        print("Val model done.")
        sess.run(tf.global_variables_initializer())
        while True:
            for n in range(len(np_sents)//8):
                sl = slice(n*8, min((n+1)*8, len(np_sents)))
                if n*8- min((n+1)*8, len(val_np_sents)) >= 0:
                    break
                s = np_sents[sl]
                t = np_targets[sl]
                tl = train_target_lens[sl]
                sl = train_in_lens[sl]
                lo, logs, _ = sess.run([train_model.loss, train_model.logits, train_model.train_op], feed_dict={
                    train_model.inputs: s,
                    train_model.targets: t,
                    train_model.target_lens: tl,
                    train_model.sequence_length: sl,
                })
                print(lo.sum())
                preds = logs.argmax(-1)
                # for p in preds:
                #     print(label_numberer.decode_sequence(p, "<STOP>"))
                print(np.equal(preds, t[:,1:]).mean())
            for n in range(len(val_np_sents//8)):
                sl = slice(n*8, min((n+1)*8, len(val_np_sents)))
                if n*8- min((n+1)*8, len(val_np_sents)) >= 0:
                    break
                s = val_np_sents[sl]
                t = val_np_targets[sl]
                tl = val_np_lens[sl]
                sl = val_np_in_lens[sl]
                preds = sess.run(val_model.sampled_ids, feed_dict={
                    val_model.inputs: s,
                    val_model.targets: t,
                    val_model.target_lens: tl,
                    val_model.sequence_length: sl,
                })
                break
                # print(np.equal(preds, t[:, 1:]).mean())
            for p in np.squeeze(preds):
                print(label_numberer.decode_sequence(t[0], "<STOP>"))
                print(label_numberer.decode_sequence(p, "<STOP>"))

                break
                
