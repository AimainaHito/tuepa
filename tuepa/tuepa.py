from numberer import Numberer
from oracle import Oracle
from states.state import State

from functools import partial
from glob import glob
from semstr.convert import FROM_FORMAT, from_text
from ucca import ioutil

class FFModel:
    def __init__(self):
        self.embeddings = tf.get_variable(name="emb", shape=[w_numberer.max, 300], dtype=tf.float32)
        self.ff = tf.layers.Dense(256, use_bias=True, activation=tf.nn.relu)
        self.proj = tf.layers.Dense(12, use_bias=False, activation=None)
        self.opt = tf.train.AdamOptimizer(0.01)

    def __call__(self, feats):
        emb = tf.reshape(tf.nn.embedding_lookup(self.embeddings,feats),[feats.shape[0],-1])
        return self.proj(self.ff(emb))

    def weights(self):
        return [self.embeddings] + self.ff.trainable_weights + self.proj.trainable_weights

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


# Marks input passages as text so that we don't accidentally train on them
def from_text_format(*args, **kwargs):
    for passage in from_text(*args, **kwargs):
        passage.extra["format"] = "text"
        yield passage


CONVERTERS = {k: partial(c, annotate=True) for k, c in FROM_FORMAT.items()}
CONVERTERS[""] = CONVERTERS["txt"] = from_text_format


def read_passages(files):
    expanded = [f for pattern in files for f in sorted(glob(pattern)) or (pattern,)]
    return ioutil.read_files_and_dirs(expanded, sentences=True, paragraphs=False,
                                      converters=CONVERTERS, lang="en")


def extract_features(state, w_numberer, train=True):
    stack = state.stack
    buffer = state.buffer
    stack_features = [w_numberer.number(e,train=train) for e in stack]
    buffer_features = [w_numberer.number(e,train=train) for e in buffer]
    return stack_features, buffer_features


if __name__ == "__main__":
    w_numberer = Numberer()
    # nt_numberer = Numberer()

    features = []
    labels = []

    max_s = max_b = -1
    for passage in read_passages(["../../train/UCCA_English-Wiki/*"]):
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

    for n,feature in enumerate(features):
        while len(feature[0]) != max_s:
            feature[0].append(0)
        while len(feature[1]) != max_b:
            feature[1].append(0)
        features[n] = feature[0]+feature[1]
    import numpy as np
    features = np.array(features)
    labels = np.array(labels)
    import tensorflow as tf
    tf.enable_eager_execution()
    m = FFModel()
    batch_size = 1024
    while True:
        for n in range(len(features) // batch_size):
            m.run_step(features[n*batch_size:(n+1)*batch_size],labels=labels[n*batch_size:(n+1)*batch_size])

