import tensorflow as tf
import numpy as np


ACTIVATION_FUNCTIONS = {
    "selu": tf.nn.selu,
    "relu": tf.nn.relu,
    "tanh": tf.nn.tanh,
    "sigmoid": tf.sigmoid,
    "gelu": lambda x: 0.5 * (1.0 + tf.erf(x / tf.sqrt(2.0))),
    "none": None,
}


class BaseModel:
    """
    Base model implementing common functionality for all neural network models
    """
    def loss(self, logits, labels):
        return tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)

    def weights(self):
        raise NotImplementedError("Weights have to be specified by a concrete model implementation")

    def save(self, file_prefix):
        tf.contrib.eager.Saver(self.weights()).save(file_prefix)

    def restore(self, file_prefix):
        tf.contrib.eager.Saver(self.weights()).restore(file_prefix)



class UpDownWithResiduals:
    def __init__(self, upsample_units, input_size, activation):
        self._input_size = input_size
        self.up = tf.layers.Dense(upsample_units, activation)
        self.down = None
        self.built = False

    def __call__(self, input):
        if not self.built:
            self.down = tf.layers.Dense(input.shape[-1], use_bias=False, activation=None)
            self.built = True
        return self.down(self.up(input)) + input

    def weights(self):
        return self.up.trainable_weights + self.down.trainable_weights

    @property
    def input_size(self):
        return self._input_size


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


EPSILON = 1e-6


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


class FFModel(BaseModel):
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

    def predict(self, feats):
        return tf.argmax(self(feats, train=False))

    def restore(self, file_prefix, args):
        # Initialize layers with an empty batch
        self(np.zeros(
            (1, args.shapes.max_stack_size + args.shapes.max_buffer_size),
            dtype=np.int32
        ))
        super().restore(file_prefix)

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


class ElModel(BaseModel):
    def __init__(self, args, feed_forward_layers, num_labels):

        self.history_embeddings = tf.get_variable(
            name="embeddings",
            shape=[num_labels, args.history_embedding_size],
            dtype=tf.float32
        )
        self.non_terminal_embedding = tf.get_variable("non_terminal", shape=[1, args.embedding_size], dtype=tf.float32)
        self.padding_embedding = tf.get_variable("padding", shape=[1, args.embedding_size], dtype=tf.float32)

        # TODO: check if CudnnGRU is available + fallback to standard tensorflow implementation
        # elmo processor rnn
        self.sentence_bi_rnn = tf.contrib.cudnn_rnn.CudnnGRU(1, args.bi_rnn_neurons, direction="bidirectional")
        self.sentence_top_rnn = tf.contrib.cudnn_rnn.CudnnGRU(1, args.top_rnn_neurons)

        # history rnn
        self.history_rnn = tf.contrib.cudnn_rnn.CudnnGRU(1, args.history_rnn_neurons)

        # dense layers
        self.feed_forward_layers = feed_forward_layers
        self.downsampling_layer = tf.layers.Dense(
            self.feed_forward_layers[0].input_size if isinstance(self.feed_forward_layers[0], UpDownWithResiduals) else
            self.feed_forward_layers[0].units, tf.nn.selu)

        self.projection_layer = tf.layers.Dense(num_labels, use_bias=False, activation=None)

        self.optimizer = tf.train.AdamOptimizer(args.learning_rate)

        self.input_dropout = args.input_dropout
        self.layer_dropout = args.layer_dropout

    """
    Processes features and outputs scores over state transitions.

    :param batch: A namedtuple holding history features, stack and buffer features, elmo embeddings, non-terminal positions,
                  padding positions and sentence and history lengths.

    Features are:
        history of transitions,
        ELMo embeddings for each word in the sentence
        words and nodes of stack and buffer

    The ELMo embeddings are processed by a bi-GRU with a stacked uni-directional GRU. The transition history by another
    GRU. The final states of both RNNs are concatenated with the stack and buffer features and fed through several dense
    layers consisting of (non-linear) up- and (linear) downsampling. The dense layers also feature residual connections.
    """

    def __call__(self, batch, train=False):
        # unpack batch
        history = batch.history_features
        features = batch.stack_and_buffer_features
        elmo = batch.elmo_embeddings

        non_terminal_positions = batch.non_terminals
        padding = batch.padding

        sentence_lengths = batch.sentence_lengths
        history_lengths = batch.history_lengths

        batch_size = features.shape[0]
        batch_indices = tf.expand_dims(tf.range(batch_size, dtype=tf.int64), 1)

        # swaps batch with time dimension
        swap_batch_with_time = lambda x: tf.transpose(x, [1, 0, 2])

        history_input = tf.nn.embedding_lookup(self.history_embeddings, history)
        # [time, batch, D]
        history_input = swap_batch_with_time(history_input)

        history_rnn_outputs, _ = self.history_rnn(history_input)
        # [batch, time, D]
        history_rnn_outputs = swap_batch_with_time(history_rnn_outputs)

        # extract final state for each sequence
        state_selectors = tf.concat([batch_indices,
                                     tf.expand_dims(history_lengths, axis=1)],
                                    axis=1)
        history_rnn_state = tf.gather_nd(history_rnn_outputs, state_selectors)

        # [time, batch, D]
        elmo = swap_batch_with_time(elmo)
        bi_rnn_outputs, _ = self.sentence_bi_rnn(elmo)

        sentence_mask = tf.expand_dims(tf.sequence_mask(sentence_lengths, bi_rnn_outputs.shape[0], dtype=tf.float32),
                                       axis=-1)
        bi_rnn_outputs *= swap_batch_with_time(sentence_mask)

        # [batch, time, D]
        top_rnn_output, _ = self.sentence_top_rnn(bi_rnn_outputs)
        top_rnn_output = swap_batch_with_time(top_rnn_output)

        # extract final state for each sequence
        state_selectors = tf.concat([batch_indices,
                                     tf.expand_dims(sentence_lengths, axis=1)],
                                    axis=1)
        top_rnn_outputs = tf.gather_nd(top_rnn_output, state_selectors)

        # add non terminal representation to stack / buffer features
        features += self.non_terminal_embedding * tf.cast(tf.expand_dims(non_terminal_positions, -1), dtype=tf.float32)
        features += self.padding_embedding * tf.cast(tf.expand_dims(padding, -1), dtype=tf.float32)
        feedforward_input = tf.reshape(
            features,
            [batch_size, -1]
        )

        feedforward_input = self.downsampling_layer(
            tf.concat([history_rnn_state, feedforward_input, top_rnn_outputs], -1))
        if train:
            feedforward_input = tf.nn.dropout(feedforward_input, self.input_dropout)

        for layer in self.feed_forward_layers:
            feedforward_input = layer(feedforward_input)
            if train:
                feedforward_input = tf.nn.dropout(feedforward_input, self.layer_dropout)

        return self.projection_layer(feedforward_input)

    def weights(self):
        return (sum((layer.weights() if isinstance(layer,UpDownWithResiduals) else layer.trainable_weights for layer in self.feed_forward_layers),
                    []) + self.downsampling_layer.trainable_weights
                + self.projection_layer.trainable_weights + self.sentence_bi_rnn.trainable_weights + [
                    self.history_embeddings]
                + self.sentence_top_rnn.trainable_weights + self.history_rnn.trainable_weights + [
                    self.non_terminal_embedding, self.padding_embedding]
                )

    def run_step(self, batch, train=False):
        labels = batch.labels
        if train:
            with tf.GradientTape() as tape:
                logits = self(batch, train=True)
                predictions = tf.to_int32(tf.argmax(logits, axis=-1))
                accuracy = tf.reduce_mean(tf.to_float(tf.equal(predictions, labels)))
                x_ent = self.loss(logits, labels=labels)

            grads = tape.gradient(x_ent, self.weights())
            self.optimizer.apply_gradients(
                zip(grads, self.weights()), global_step=tf.train.get_or_create_global_step()
            )

        else:
            logits = self(batch)
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


class ResidualFeedforward(tf.keras.layers.Layer):
    def __init__(self, attention_neurons, feedforward_neurons, **kwargs):
        super().__init__(self, **kwargs)
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


class TransformerModel(BaseModel):
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

        for layer in self.encoder:
            embeddings = layer(embeddings)

        return self.projection_layer(embeddings[:, 0])

    def weights(self):
        return (
                [self.embeddings, self.position_embeddings]
                + sum((layer.trainable_weights for layer in self.encoder), [])
        )

    def run_step(self, feats, labels, train=False):
        if train:
            with tf.GradientTape() as tape:
                logits = self(feats)
                predictions = tf.to_int32(tf.argmax(logits, axis=-1))
                accuracy = tf.reduce_mean(tf.to_float(tf.equal(predictions, labels)))
                x_ent = self.loss(logits, labels=labels)

            gradients = tape.gradient(x_ent, self.weights())
            self.optimizer.apply_gradients(zip(gradients, self.weights()),
                                           global_step=tf.train.get_or_create_global_step())

        else:
            logits = self(feats)
            predictions = tf.to_int32(tf.argmax(logits, axis=-1))
            accuracy = tf.reduce_mean(tf.to_float(tf.equal(predictions, labels)))
            x_ent = self.loss(logits, labels=labels)

        return sum(x_ent.numpy()), accuracy

    def restore(self, file_prefix, args):
        # Initialize layers with an empty batch
        self(np.zeros(
            (1, args.shapes.max_stack_size + args.shapes.max_buffer_size + args.max_training_length + 1),
            dtype=np.int32
        ))
        super().restore(file_prefix)


    def build_encoder(self):
        encoder = []
        for _ in range(self.layers):
            encoder.append(SelfAttention(self.args.num_heads, self.args.embedding_size))
            encoder.append(ResidualFeedforward(self.args.embedding_size, self.args.self_attention_neurons))

        return encoder
