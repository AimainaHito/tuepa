from numberer import Numberer
from oracle import Oracle
from states.state import State

from functools import partial
from glob import glob
from semstr.convert import FROM_FORMAT, from_text
from ucca import ioutil

import tensorflow as tf


def split_heads(t, hidden_size, num_heads):
    """
    Splits the last dimension of `t` into `self.config.num_heads` dimensions.
    """
    batch_size = tf.shape(t)[0]  # batch dimension
    batch_length = tf.shape(t)[1]  # length dimension
    head_size = hidden_size // num_heads  # dimensions per head
    # transpose to [batch, num_heads, length, head] so that attention is across length dimension
    return tf.transpose(tf.reshape(t, [batch_size, batch_length, num_heads, head_size]),
                        [0, 2, 1, 3])


class FFModel:
    def __init__(self):
        self.embeddings = tf.get_variable(name="emb", shape=[w_numberer.max, 300], dtype=tf.float32)
        self.ff = tf.layers.Dense(256, use_bias=True, activation=tf.nn.relu)
        self.ff2 = tf.layers.Dense(256, use_bias=True, activation=tf.nn.relu)
        self.proj = tf.layers.Dense(12, use_bias=False, activation=None)
        self.opt = tf.train.AdamOptimizer(0.01)

    def __call__(self, feats):
        emb = tf.reshape(tf.nn.embedding_lookup(self.embeddings,feats),[feats.shape[0],-1])
        return self.proj(self.ff2(self.ff(emb)))

    def weights(self):
        return [self.embeddings] + self.ff.trainable_weights + self.proj.trainable_weights + self.ff2.trainable_weights

    def loss(self, logits, labels):
        return tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=labels)

    def run_step(self, feats, labels):
        with tf.GradientTape() as tape:
            logits = self(feats)
            predictions = tf.to_int32(tf.argmax(logits,axis=-1))
            acc = tf.reduce_mean(tf.to_float(tf.equal(predictions,labels)))
            x_ent = self.loss(logits,labels=labels)
        grads = tape.gradient(x_ent,self.weights())
        self.opt.apply_gradients(zip(grads,self.weights()),global_step=tf.train.get_or_create_global_step())
        print(sum(x_ent.numpy()), acc)

class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.built = False
        self.num_heads = num_heads

    def compute_output_shape(self, input_shape):
        return input_shape

    def build(self,input_shape):
        self.q = tf.layers.Dense(input_shape[-1], name="q", use_bias=False)
        self.k = tf.layers.Dense(input_shape[-1], name="k", use_bias=False)
        self.v = tf.layers.Dense(input_shape[-1], name="v", use_bias=False)

        self.built = True


    def trainable_weights(self):
        return self.q.trainable_weights+self.k.trainable_weights+self.v.trainable_weights

    def __call__(self, inputs, **kwargs):
        if not self.built:
            self.build(inputs.shape)

        queries = self.q(inputs)

        keys = self.k(inputs)

        values = self.v(inputs)

        queries = split_heads(queries,hidden_size=256, num_heads=8)
        keys = split_heads(keys,hidden_size=256, num_heads=8)
        values = split_heads(values,hidden_size=256, num_heads=8)

        # dot product + scale queries
        scale = tf.constant(8 ** -0.5)
        scores = tf.matmul(queries * scale, keys, transpose_b=True)

        # padding hack part one (hopefully there's a better way of doing this)
        # # set all pad positions for the last two dimensions to max negative float value
        # mask_s = tf.to_float(tf.expand_dims(tf.expand_dims(
        #     tf.logical_not(tf.sequence_mask(self.seq_lens, tf.shape(inputs)[1])),
        #     1), -1)) * -1e38
        # mask_t = tf.to_float(tf.expand_dims(tf.expand_dims(
        #     tf.logical_not(tf.sequence_mask(self.seq_lens, tf.shape(inputs)[1])),
        #     1), 1)) * -1e38
        # scores += mask_s + mask_t

        # remember non-pad positions
        # non_pad_ids = tf.to_int32(tf.where(scores > -1e30))

        # attention_dropout = tf.layers.Dropout(1 - 0.1)
        scores = tf.nn.softmax(scores+0.0001)

        # padding hack part two
        # 1. get all non pad positions from scores
        # 2. create new tensor with non pad items and shape of scores
        # scores = tf.scatter_nd(non_pad_ids, tf.gather_nd(scores, non_pad_ids), tf.shape(scores))
        # self.scores.append(scores)

        # apply scores to values
        heads = tf.matmul(scores, values)

        # restore [batch, length, num_heads, head] order and rejoin num_heads and head
        heads = tf.reshape(tf.transpose(heads, [0, 2, 1, 3]), (tf.shape(inputs)[0], tf.shape(inputs)[1],
                                                               tf.shape(inputs)[-1]))

        proj = tf.layers.Dense(256, use_bias=False)
        return proj(heads) + inputs


class TransformerModel:
    def __init__(self):
        self.embeddings = tf.get_variable(name="emb", shape=[w_numberer.max, 256], dtype=tf.float32)
        self.pos_embeds = tf.get_variable("pos_embeds", shape=[250, 256])
        self.opt = tf.train.AdamOptimizer(0.001)
        self.layers = 2
        self.encoder = self.build_encoder()
        self.proj = tf.layers.Dense(12,use_bias=False)

    def __call__(self, feats):
        emb = tf.nn.embedding_lookup(self.embeddings, feats)
        emb += tf.nn.embedding_lookup(self.pos_embeds, tf.clip_by_value(tf.range(0, tf.shape(emb)[1]),
                                                                        clip_value_min=0,
                                                                        clip_value_max=250))
        for layer in self.encoder:
            emb = layer(emb)

        return self.proj(emb[:,0])

    def weights(self):
        return [self.embeddings, self.pos_embeds] + self.encoder[0].trainable_weights() + self.encoder[
            1].trainable_weights + self.encoder[2].trainable_weights + self.encoder[3].trainable_weights() + \
               self.encoder[4].trainable_weights + self.encoder[5].trainable_weights

    def loss(self, logits, labels):
        return tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=labels)

    def run_step(self, feats, labels):
        with tf.GradientTape() as tape:
            logits = self(feats)
            predictions = tf.to_int32(tf.argmax(logits,axis=-1))
            acc = tf.reduce_mean(tf.to_float(tf.equal(predictions,labels)))
            x_ent = self.loss(logits,labels=labels)
        grads = tape.gradient(x_ent,self.weights())
        self.opt.apply_gradients(zip(grads,self.weights()),global_step=tf.train.get_or_create_global_step())
        print(sum(x_ent.numpy()), acc)

    def build_encoder(self):
        encoder = []
        for n in range(self.layers):
            encoder.append(SelfAttention(4))
            encoder.append(tf.layers.Dense(512,tf.nn.relu))
            encoder.append(tf.layers.Dense(256,use_bias=False))
        return encoder



# Marks input passages as text so that we don't accidentally train on them
def from_text_format(*args, **kwargs):
    for passage in from_text(*args, **kwargs):
        passage.extra["format"] = "text"
        yield passage


CONVERTERS = {k: partial(c, annotate=True) for k, c in FROM_FORMAT.items()}
CONVERTERS[""] = CONVERTERS["txt"] = from_text_format


def read_passages(files):
    print(files)
    expanded = [f for pattern in files for f in sorted(glob(pattern)) or (pattern,)]
    print(expanded)
    return ioutil.read_files_and_dirs(expanded, sentences=True, paragraphs=False,
                                      converters=CONVERTERS, lang="en")


def extract_features(state, w_numberer, train=True):
    stack = state.stack
    buffer = state.buffer
    stack_features = [w_numberer.number(e,train=train) for e in stack]
    buffer_features = [w_numberer.number(e,train=train) for e in buffer]
    return stack_features, buffer_features


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python tuepa.py <PATH>")
        sys.exit(1)
    path = sys.argv[1]

    w_numberer = Numberer(first_elements=["<PAD>","<SEP>"])
    # nt_numberer = Numberer()

    features = []
    labels = []

    max_s = max_b = -1
    for passage in read_passages([path]):
        s = State(passage)
        o = Oracle(passage)
        while not s.finished:
            actions = o.generate_actions(state=s)
            a = next(actions)
            stack_features, buffer_features = extract_features(s, w_numberer)
            label = a.type_id
            s.transition(a)
            a.apply()
            features.append((stack_features,buffer_features))
            max_s = max(len(stack_features), max_s)
            max_b = max(len(buffer_features), max_b)
            labels.append(label)
        if len(features) > 250000:
            break
    print(len(features))

    for n,feature in enumerate(features):
        features[n] = feature[0]+[1]+feature[1]

    import numpy as np
    # features = np.array(features)
    labels = np.array(labels)
    import tensorflow as tf
    tf.enable_eager_execution()
    m = TransformerModel()
    batch_size = 256
    print(w_numberer.max)
    import random

    while True:
        pairs = list(zip(features, labels))
        random.shuffle(pairs)
        features, labels = zip(*pairs)
        for n in range(len(features) // batch_size):
            batch = features[n*batch_size:(n+1)*batch_size]
            max_l = max(map(len,batch))
            for u, feature in enumerate(batch):
                while len(batch[u]) < max_l:
                    batch[u].append(0)
            m.run_step(np.array(batch),labels=labels[n*batch_size:(n+1)*batch_size])

