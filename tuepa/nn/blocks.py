class UpDownWithResiduals(tf.keras.layers.Layer):
    def __init__(self, upsample_units, input_size, activation, **kwargs):
        super(UpDownWithResiduals, self).__init__(**kwargs)
        self._input_size = input_size
        self.up = tf.layers.Dense(upsample_units, activation)
        self.down = tf.layers.Dense(
            input_size, use_bias=False, activation=None)
        self._trainable_weights = self.up.trainable_weights + self.down.trainable_weights

    def __call__(self, input):
        return self.down(self.up(input)) + input

    @property
    def trainable_weights(self):
        return self._trainable_weights

    @property
    def weights(self):
        return self.up.trainable_weights + self.down.trainable_weights

    @property
    def input_size(self):
        return self._input_size


ACTIVATION_FUNCTIONS = {
    "selu": tf.nn.selu,
    "relu": tf.nn.relu,
    "tanh": tf.nn.tanh,
    "sigmoid": tf.sigmoid,
    "gelu": lambda x: 0.5 * (1.0 + tf.erf(x / tf.sqrt(2.0))),
    "none": None,
}



"""
Creates Feedforward layers from a json list
Example:
[{"neurons" : 512, "activation" : "relu"}, {"neurons" : 512, "activation" : "relu", "updown" : true}]

:param json_data: A list where where each entry is a dictionary containing
"neurons" (integer), "bias" (boolean, default is True), "activation"
(one of "relu", "sigmoid", "tanh" or "none" default is "none") and "updown" (boolean, default is False)

:return: A list of tensorflow Dense layers
"""


def feed_forward_from_json(json_data):
    layers = []

    for layer in json_data:
        layers.append(
            UpDownWithResiduals(
                upsample_units=layer["neurons"] * 2,
                input_size=layer["neurons"],
                activation=ACTIVATION_FUNCTIONS[layer['activation']]
            ) if layer.get('updown', False) else tf.layers.Dense(
                layer["neurons"],
                use_bias=layer.get("bias", True),
                activation=ACTIVATION_FUNCTIONS.get(layer["activation"], None)
            )
        )

    return layers

class LayerNorm(tf.keras.layers.Layer):
    """
    Layer normalization layer as in https://github.com/tensorflow/models/blob/master/official/transformer/model/transformer.py
    """

    def __init__(self, hsize, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hsize
        self.scale = tf.get_variable("{}/scale".format(self.name), [self.hidden_size],
                                     initializer=tf.ones_initializer())
        self.bias = tf.get_variable("{}/bias".format(self.name), [self.hidden_size],
                                    initializer=tf.zeros_initializer())

    def __call__(self, x, **kwargs):
        mean, variance = tf.nn.moments(x, axes=[-1], keep_dims=True)
        norm_x = (x - mean) / tf.rsqrt(variance + EPSILON)
        return norm_x * self.scale + self.bias

    def weights(self):
        return [self.scale, self.bias]


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
    def __init__(self, num_heads, neurons, **kwargs):
        super().__init__(**kwargs)
        self.built = False
        self.neurons = neurons
        self.num_heads = num_heads
        self.projection_layer = tf.layers.Dense(neurons, use_bias=False)

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

        queries = split_heads(
            queries, hidden_size=self.neurons, num_heads=self.num_heads)
        keys = split_heads(keys, hidden_size=self.neurons,
                           num_heads=self.num_heads)
        values = split_heads(
            values, hidden_size=self.neurons, num_heads=self.num_heads)

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


class ResidualFeedforward(tf.keras.layers.Layer):
    def __init__(self, attention_neurons, feedforward_neurons, **kwargs):
        super().__init__(self, **kwargs)
        self.feedforward = tf.layers.Dense(feedforward_neurons, tf.nn.relu)
        self.projection_layer = tf.layers.Dense(
            attention_neurons, use_bias=False)
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

