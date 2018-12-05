import tensorflow as tf


ACTIVATION_FUNCTIONS = {
    "relu" : tf.nn.relu,
    "tanh" : tf.nn.tanh,
    "sigmoid" : tf.sigmoid,
    "none" : None,
}

"""
Creates Feedforward layers from a json list
Example:
[{"neurons" : 512, "activation" : "relu", "neurons" : 512, "activation" : "relu"}]

:param json_data: A list where where each entry is a dictionary containing
"neurons" (integer), "bias" (boolean, default is True) and "activation"
(one of "relu", "sigmoid", "tanh" or "none" default is "none")

:return: A list of tensorflow Dense layers
"""
def feed_forward_from_json(json_data):
    layers = []

    for layer in json_data:
        layers.append(tf.layers.Dense(
            layer["neurons"],
            use_bias=layer.get("bias", True),
            activation=ACTIVATION_FUNCTIONS.get(layer["activation"], None)
        ))

    return layers


class FFModel:
    def __init__(self, dictionary_size, embedding_dims, feed_forward_layers):
        self.embeddings = tf.get_variable(
            name="embeddings",
            shape=[dictionary_size, embedding_dims],
            dtype=tf.float32
        )

        self.feed_forward_layers = feed_forward_layers
        self.projection_layer = tf.layers.Dense(12, use_bias=False, activation=None)
        self.optimizer = tf.train.AdamOptimizer(0.01)

    def __call__(self, features):
        layer_input = tf.reshape(
            tf.nn.embedding_lookup(self.embeddings, features),
            [features.shape[0], -1]
        )
        for layer in self.feed_forward_layers:
            layer_input = layer(layer_input)

        return self.projection_layer(layer_input)

    def weights(self):
        return (
            [self.embeddings]
            + sum((layer.trainable_weights for layer in self.feed_forward_layers), [])
            + self.projection_layer.trainable_weights
        )

    def loss(self, logits, labels):
        return tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)

    def run_step(self, feats, labels, train=False):
        if train:
            with tf.GradientTape() as tape:
                logits = self(feats)
                predictions = tf.to_int32(tf.argmax(logits, axis=-1))
                accuracy = tf.reduce_mean(tf.to_float(tf.equal(predictions, labels)))
                x_ent = self.loss(logits, labels=labels)

            grads = tape.gradient(x_ent, self.weights())
            self.optimizer.apply_gradients(
                zip(grads, self.weights()), global_step=tf.train.get_or_create_global_step()
            )

        else:
            logits = self(feats)
            predictions = tf.to_int32(tf.argmax(logits, axis=-1))
            accuracy = tf.reduce_mean(tf.to_float(tf.equal(predictions, labels)))
            x_ent = self.loss(logits, labels=labels)

        return sum(x_ent.numpy()), accuracy
