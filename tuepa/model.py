import tensorflow as tf

ACTIVATION_FUNCTIONS = {
    "relu": tf.nn.relu,
    "tanh": tf.nn.tanh,
    "sigmoid": tf.sigmoid,
    "none": None,
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


EPSILON = 1e-6


class LayerNorm(tf.layers.Layer):
    """
    Layer normalization layer as in https://github.com/tensorflow/models/blob/master/official/transformer/model/transformer.py
    """

    def __init__(self, hsize, **kwargs):
        super(LayerNorm, self).__init__(**kwargs)
        self.hidden_size = hsize
        self.scale = tf.get_variable("scale", [self.hidden_size],
                                     initializer=tf.ones_initializer())
        self.bias = tf.get_variable("bias", [self.hidden_size],
                                    initializer=tf.zeros_initializer())

    def __call__(self, x, **kwargs):
        mean, variance = tf.nn.moments(x, axes=[-1], keep_dims=True)
        norm_x = (x - mean) / tf.rsqrt(variance + EPSILON)
        return norm_x * self.scale + self.bias

    def weights(self):
        return [self.scale] + [self.bias]


class FFModel:
    def __init__(
            self, dictionary_size, embedding_dims, feed_forward_layers, num_labels,
            initial_learning_rate=0.01, input_dropout=1, layer_dropout=1
    ):
        self.embeddings = tf.get_variable(
            name="embeddings",
            shape=[dictionary_size, embedding_dims],
            dtype=tf.float32
        )

        self.feed_forward_layers = feed_forward_layers
        self.projection_layer = tf.layers.Dense(num_labels, use_bias=False, activation=None)
        self.optimizer = tf.train.AdamOptimizer(initial_learning_rate)
        self.input_dropout = input_dropout
        self.layer_dropout = layer_dropout

    def __call__(self, features, train=False):
        layer_input = tf.reshape(
            tf.nn.embedding_lookup(self.embeddings, features),
            [features.shape[0], -1]
        )
        if train:
            layer_input = tf.nn.dropout(layer_input, self.input_dropout)

        for layer in self.feed_forward_layers:
            layer_input = layer(layer_input)
            if train:
                layer_input = tf.nn.dropout(layer_input, self.layer_dropout)

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
                logits = self(feats, train=True)
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


def split_heads(t, hidden_size, num_heads):
    """
    Splits the last dimension of `t` into `self.config.num_heads` dimensions.
    """
    batch_size = tf.shape(t)[0]  # Batch dimension
    batch_length = tf.shape(t)[1]  # Length dimension
    head_size = hidden_size // num_heads  # Dimensions per head
    # Transpose to [batch, num_heads, length, head] so that attention is across length dimension
    return tf.transpose(tf.reshape(t, [batch_size, batch_length, num_heads, head_size]),
                        [0, 2, 1, 3])


class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, neurons, scope, **kwargs):
        super().__init__(**kwargs)
        self.built = False
        self.neurons = neurons
        self.num_heads = num_heads
        self.projection_layer = tf.layers.Dense(neurons, use_bias=False)
        self.scope = scope

    def compute_output_shape(self, input_shape):
        return input_shape

    def build(self, input_shape):
        self.q = tf.layers.Dense(input_shape[-1], name="q", use_bias=False)
        self.k = tf.layers.Dense(input_shape[-1], name="k", use_bias=False)
        self.v = tf.layers.Dense(input_shape[-1], name="v", use_bias=False)
        self.layer_norm = LayerNorm(input_shape[-1])
        self.built = True

    @property
    def trainable_weights(self):
        return self.q.trainable_weights + self.k.trainable_weights + self.v.trainable_weights + self.projection_layer.trainable_weights + self.layer_norm.weights()

    def __call__(self, inputs, **kwargs):
        if not self.built:
            self.build(inputs.shape)

        queries = self.q(inputs)
        keys = self.k(inputs)
        values = self.v(inputs)

        queries = split_heads(queries, hidden_size=self.neurons, num_heads=self.num_heads)
        keys = split_heads(keys, hidden_size=self.neurons, num_heads=self.num_heads)
        values = split_heads(values, hidden_size=self.neurons, num_heads=self.num_heads)

        # Dot product + scale queries
        scale = tf.constant(8 ** -0.5)
        scores = tf.matmul(queries * scale, keys, transpose_b=True)

        # Attention_dropout = tf.layers.Dropout(1 - 0.1)
        scores = tf.nn.softmax(scores + 0.0001)

        # Apply scores to values
        heads = tf.matmul(scores, values)

        # Restore [batch, length, num_heads, head] order and rejoin num_heads and head
        heads = tf.reshape(tf.transpose(heads, [0, 2, 1, 3]), (tf.shape(inputs)[0], tf.shape(inputs)[1],
                                                               tf.shape(inputs)[-1]))

        # Apply projection layer and layer normalization
        return self.layer_norm(self.projection_layer(heads) + inputs)


class ResidualFeedforward:
    def __init__(self, attention_neurons, feedforward_neurons, scope):
        self.scope = scope
        self.feedforward = tf.layers.Dense(feedforward_neurons, tf.nn.relu)
        self.projection_layer = tf.layers.Dense(attention_neurons, use_bias=False)
        self.layer_norm = LayerNorm(attention_neurons)
    @property
    def trainable_weights(self):
        return self.feedforward.trainable_weights + self.projection_layer.trainable_weights + self.layer_norm.weights()

    def __call__(self, inputs):
        output = self.feedforward(inputs)
        # Normalized feedforward output with residual connections
        return self.layer_norm(
            self.projection_layer(output)
            + inputs
        )


class TransformerModel:
    def __init__(self, dictionary_size, num_labels, args):
        self.embeddings = tf.get_variable(name="embeddings", shape=[dictionary_size, args.embedding_size],
                                          dtype=tf.float32)
        self.position_embeddings = tf.get_variable("position_embeddings",
                                                   shape=[args.max_positions, args.embedding_size])
        self.optimizer = tf.train.AdamOptimizer(args.learning_rate)
        self.args = args
        self.layers = args.num_layers
        self.encoder = self.build_encoder()
        self.projection_layer = tf.layers.Dense(num_labels, use_bias=False)

    def __call__(self, feats):
        embeddings = tf.nn.embedding_lookup(self.embeddings, feats)
        embeddings += tf.nn.embedding_lookup(
            self.position_embeddings,
            tf.clip_by_value(tf.range(0, tf.shape(embeddings)[1]), clip_value_min=0,
                             clip_value_max=self.args.max_positions)
        )

        for n, layer in enumerate(self.encoder):
            embeddings = layer(embeddings)

        return self.projection_layer(embeddings[:, 0])

    def weights(self):
        return (
                [self.embeddings, self.position_embeddings]
                + sum((layer.trainable_weights for layer in self.encoder), [])
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

            gradients = tape.gradient(x_ent, self.weights())
            self.optimizer.apply_gradients(zip(gradients,self.weights()),
                                           global_step=tf.train.get_or_create_global_step())

        else:
            logits = self(feats)
            predictions = tf.to_int32(tf.argmax(logits, axis=-1))
            accuracy = tf.reduce_mean(tf.to_float(tf.equal(predictions, labels)))
            x_ent = self.loss(logits, labels=labels)

        return sum(x_ent.numpy()), accuracy

    def build_encoder(self):
        encoder = []
        for n in range(self.layers):
            encoder.append(SelfAttention(self.args.num_heads, self.args.embedding_size, "attn{}".format(n)))
            encoder.append(
                ResidualFeedforward(self.args.embedding_size, self.args.self_attention_neurons, "ff{}".format(n)))

        return encoder
