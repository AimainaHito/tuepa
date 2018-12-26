import tensorflow as tf

ACTIVATION_FUNCTIONS = {
    "selu": tf.nn.selu,
    "relu": tf.nn.relu,
    "tanh": tf.nn.tanh,
    "sigmoid": tf.sigmoid,
    "gelu": lambda x: 0.5 * (1.0 + tf.erf(x / tf.sqrt(2.0))),
    "none": None,
}


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
[{"neurons" : 512, "activation" : "relu", "neurons" : 512, "activation" : "relu"}]

:param json_data: A list where where each entry is a dictionary containing
"neurons" (integer), "bias" (boolean, default is True) and "activation"
(one of "relu", "sigmoid", "tanh" or "none" default is "none")

:return: A list of tensorflow Dense layers
"""


def feed_forward_from_json(json_data):
    layers = []

    for layer in json_data:
        layers.append(
            UpDownWithResiduals(
                upsample_units=layer["neurons"] * 2,
                input_size=layer["neurons"],
                activation=ACTIVATION_FUNCTIONS[layer['activation']]))

    return layers


class ElModel:
    def __init__(
            self, args, feed_forward_layers, num_labels, word_embedding_dim):

        self.history_embeddings = tf.get_variable(
            name="embeddings",
            shape=[num_labels, args.embedding_size],
            dtype=tf.float32
        )
        self.non_terminal_embedding = tf.get_variable("non_terminal", shape=[1, word_embedding_dim], dtype=tf.float32)
        self.padding_embedding = tf.get_variable("padding", shape=[1, word_embedding_dim], dtype=tf.float32)

        # TODO: check if CudnnGRU is available + fallback to standard tensorflow implementation
        # elmo processor rnn
        self.sentence_bi_rnn = tf.contrib.cudnn_rnn.CudnnGRU(1, args.bi_rnn_neurons, direction="bidirectional")
        self.sentence_top_rnn = tf.contrib.cudnn_rnn.CudnnGRU(1, args.top_rnn_neurons)

        # history rnn
        self.history_rnn = tf.contrib.cudnn_rnn.CudnnGRU(1, args.history_rnn_neurons)

        # dense layers
        self.feed_forward_layers = feed_forward_layers
        self.downsampling_layer = tf.layers.Dense(self.feed_forward_layers[0].input_size, tf.nn.selu)
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
        batch_indices = tf.expand_dims(tf.range(batch_size,dtype=tf.int64), 1)

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
                                     tf.expand_dims(history_lengths,axis=1)],
                                    axis=1)
        history_rnn_state = tf.gather_nd(history_rnn_outputs, state_selectors)

        # [time, batch, D]
        elmo = swap_batch_with_time(elmo)
        bi_rnn_outputs, _ = self.sentence_bi_rnn(elmo)

        sentence_mask = tf.expand_dims(tf.sequence_mask(sentence_lengths, bi_rnn_outputs.shape[0], dtype=tf.float32), axis=-1)
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
        features += self.non_terminal_embedding * tf.cast(tf.expand_dims(non_terminal_positions,-1),dtype=tf.float32)
        features += self.padding_embedding * tf.cast(tf.expand_dims(padding,-1),dtype=tf.float32)
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
        return (sum((layer.weights() for layer in self.feed_forward_layers),
                    []) + self.downsampling_layer.trainable_weights
                + self.projection_layer.trainable_weights + self.sentence_bi_rnn.trainable_weights + [
                    self.history_embeddings]
                + self.sentence_top_rnn.trainable_weights + self.history_rnn.trainable_weights + [
                    self.non_terminal_embedding, self.padding_embedding]
                )

    def loss(self, logits, labels):
        return tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)

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
