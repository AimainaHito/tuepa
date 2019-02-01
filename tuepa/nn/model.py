from .base_model import BaseModel
from .blocks import *

import tensorflow as tf
import numpy as np
import json

import opennmt as onmt


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
        self.projection_layer = tf.layers.Dense(
            num_labels, use_bias=False, activation=None)
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
                accuracy = tf.reduce_mean(
                    tf.to_float(tf.equal(predictions, labels)))
                x_ent = self.loss(logits, labels=labels)

            grads = tape.gradient(x_ent, self.weights())
            self.optimizer.apply_gradients(
                zip(grads, self.weights()), global_step=tf.train.get_or_create_global_step()
            )
        else:
            logits = self(feats)
            predictions = tf.to_int32(tf.argmax(logits, axis=-1))
            accuracy = tf.reduce_mean(
                tf.to_float(tf.equal(predictions, labels)))
            x_ent = self.loss(logits, labels=labels)

        return sum(x_ent.numpy()), accuracy


def get_timing_signal_1d(positions,
                         channels,
                         min_timescale=1.0,
                         max_timescale=1.0e4,
                         start_index=1):
    """
    Taken from tensor2tensor git repository,slightly modified.
    """
    import math
    position = tf.to_float(positions + start_index)
    num_timescales = channels // 2
    log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            tf.maximum(tf.to_float(num_timescales) - 1, 1))
    inv_timescales = min_timescale * tf.exp(
        tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
    scaled_time = tf.expand_dims(position, -1) * inv_timescales
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=-1)
    return signal


class ElModel(BaseModel):
    def __init__(self, args, num_labels, num_dependencies, num_pos, num_ner, train, predict=False,batch=None):
        super().__init__()
        self.args = args

        feature_tokens = self.args.shapes.max_buffer_size + self.args.shapes.max_stack_size

        # Inputs
        if not predict:
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
            self.height = tf.placeholder(name="height", shape=[None, feature_tokens], dtype=tf.int32)
            self.inc = tf.placeholder(name="inc", shape=[None, feature_tokens, self.args.num_edges], dtype=tf.int32)
            self.out = tf.placeholder(name="out", shape=[None, feature_tokens, self.args.num_edges], dtype=tf.int32)
            self.history = tf.placeholder(name="hist", shape=[None, None], dtype=tf.int32)

            self.sentence_lengths = tf.placeholder(name="sent_lens", shape=[None], dtype=tf.int32)
            self.history_lengths = tf.placeholder(name="hist_lens", shape=[None], dtype=tf.int32)
            self.action_ratios = tf.placeholder(name="action_ratios", shape=[None], dtype=tf.float32)
            self.node_ratios = tf.placeholder(name="node_ratios", shape=[None], dtype=tf.float32)
            self.action_counts = tf.placeholder(name="act_counts", shape=[None, self.args.num_labels],
                                                dtype=tf.int32)
            self.root = tf.placeholder(name="root", shape=[None, feature_tokens], dtype=tf.int32)
            self.elmo = tf.placeholder(name="elmo", shape=[None, None, 1324], dtype=tf.float32)
        else:
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
            ) = batch

        if not predict:
            self.labels = tf.placeholder(name="labels", shape=[None], dtype=tf.int32)

        # Embeddings
        self.history_embeddings = tf.get_variable(
            name="embeddings",
            shape=[num_labels, args.history_embedding_size],
            dtype=tf.float32
        )

        self.pos_embeddings = tf.get_variable(
            name="pos_embeddings",
            shape=[num_pos, args.embedding_size],
            dtype=tf.float32
        )

        self.dep_embeddings = tf.get_variable(
            name="dep_embeddings",
            shape=[num_dependencies, args.embedding_size],
            dtype=tf.float32
        )

        self.ner_embeddings = tf.get_variable(
            name="ner_embeddings",
            shape=[num_ner, args.embedding_size],
            dtype=tf.float32
        )

        self.edge_embeddings = tf.get_variable(
            name="edge_embeddings",
            shape=[args.num_edges, args.embedding_size],
            dtype=tf.float32)

        self.action_count_embeddings = tf.get_variable(
            name="ac_count",
            shape=[1, num_labels, max(self.args.embedding_size // 5, 10)],
            dtype=tf.float32)

        self.height_embeddings = tf.get_variable(
            name="height",
            shape=[1, feature_tokens, max(self.args.embedding_size // 5, 10)],
            dtype=tf.float32)

        self.incoming_embedding = tf.get_variable(
            name="inc",
            shape=[1, feature_tokens, args.num_edges, max(self.args.embedding_size // 5, 10)],
            dtype=tf.float32
        )
        self.out_embedding = tf.get_variable(
            name="out",
            shape=[1, feature_tokens, args.num_edges, max(self.args.embedding_size // 5, 10)],
            dtype=tf.float32
        )
        self.non_terminal_embedding = tf.get_variable("non_terminal", shape=[1, 1, self.elmo.shape[-1]],
                                                      dtype=tf.float32)
        self.padding_embedding = tf.get_variable("padding", shape=[1, 1, self.elmo.shape[-1]], dtype=tf.float32)

        self.input_dropout = args.input_dropout
        self.layer_dropout = args.layer_dropout

        # Weights
        # self.elmo_weights = tf.get_variable("weights_scale", initializer=[[[[0.5],[0.5],[0.5]]]], dtype=tf.float32)
        # self.elmo_scale = tf.get_variable("elmo_scale", initializer=1., dtype=tf.float32)
        # elmo = tf.reduce_sum(self.elmo * tf.nn.softmax(self.elmo_weights), axis=-2)
        self.history_rnn = tf.layers.Dense(args.history_rnn_neurons, activation=tf.nn.relu)
        # dense layers
        self.feed_forward_layers = feed_forward_from_json(json.loads(args.layers))
        self.child_processing_layers = tf.layers.Dense(self.args.top_rnn_neurons, activation=tf.nn.relu)
        self.projection_layer = tf.layers.Dense(num_labels, use_bias=False)
        # Utility
        batch_size = tf.shape(self.form_indices)[0]
        batch_indices = tf.expand_dims(tf.range(batch_size, dtype=tf.int32), 1)
        conc = lambda x, y: tf.concat([x, y], -1)
        #
        action_counts = tf.tile(self.action_count_embeddings, [batch_size, 1, 1])
        action_counts = conc(action_counts,
                             get_timing_signal_1d(self.action_counts, self.action_count_embeddings.shape[-1].value))
        inc = tf.tile(self.incoming_embedding, [batch_size, 1, 1, 1])
        inc = conc(inc, get_timing_signal_1d(self.inc, self.incoming_embedding.shape[-1].value))
        out = tf.tile(self.out_embedding, [batch_size, 1, 1, 1])
        out = conc(out, get_timing_signal_1d(self.inc, self.out_embedding.shape[-1].value))
        height = tf.tile(self.height_embeddings, [batch_size, 1, 1])
        height = conc(height, get_timing_signal_1d(self.height, self.height_embeddings.shape[-1].value))
        
        action_counts = tf.reshape(action_counts, [batch_size, self.args.num_labels * 10 * 2])
        inc = tf.reshape(inc, [batch_size, feature_tokens, self.args.num_edges * 10 * 2])
        out = tf.reshape(out, [batch_size, feature_tokens, self.args.num_edges * 10 * 2])
        height = tf.reshape(height, [batch_size, feature_tokens, 10 * 2])
        # elmo, state = self.elmo_lstm(inputs=switch(self.elmo * self.elmo_scale),sequence_lengths=self.sentence_lengths)
        # elmo = switch(elmo)
        # prepend padding and non terminal embedding, non-terminals + padded positions on the stack have form index
        # 0 and 1, the rest is offset by 2
        top_rnn_output = tf.concat(
            [tf.tile(self.padding_embedding, [batch_size, 1, 1]),
             tf.tile(self.non_terminal_embedding, [batch_size, 1, 1]), self.elmo],
            1)

        # extract embeddings for stack + buffer tokens
        form_features = self.extract_vectors_3d(first_d=batch_indices,
                                                second_d=self.form_indices,
                                                batch_size=batch_size,
                                                n=feature_tokens, t=top_rnn_output, lengths=self.sentence_lengths + 2)

        head_features = self.extract_vectors_3d(first_d=batch_indices,
                                                second_d=self.head_indices,
                                                batch_size=batch_size,
                                                n=feature_tokens, t=top_rnn_output, lengths=self.sentence_lengths + 2)
        child_features = self.extract_node_children(batch_size, self.child_indices, feature_tokens,
                                                    top_rnn_output, child_ids=self.child_ids,
                                                    segment_ids=self.batch_ind, lengths=self.sentence_lengths + 2)

        pos_features = tf.nn.embedding_lookup(self.pos_embeddings, self.pos)
        dep_features = tf.nn.embedding_lookup(self.dep_embeddings, self.dep_types)
        ner_features = tf.nn.embedding_lookup(self.ner_embeddings, self.ner)

        features = tf.concat(
            [form_features, head_features, child_features,],
#             , pos_features, dep_features, ner_features, inc, out, height, tf.expand_dims(tf.to_float(self.root), -1)],
            axis=-1)

        if train:
            features = tf.nn.dropout(features, self.input_dropout)

        feedforward_input = tf.reshape(features, [batch_size, features.shape[1] * features.shape[2]])

        history_input = tf.nn.embedding_lookup(self.history_embeddings, self.history)
        history_input = tf.reshape(history_input, [batch_size, 10 * self.history_embeddings.shape[-1]])

        feature_vec = tf.concat([history_input,feedforward_input,action_counts, tf.expand_dims(self.action_ratios, -1),
                                 tf.expand_dims(self.node_ratios, -1)], -1)

        if train:
            feature_vec = tf.nn.dropout(feature_vec, self.input_dropout)

        for layer in self.feed_forward_layers:
            feature_vec = layer(feature_vec)
            if train:
                feature_vec = tf.nn.dropout(feature_vec, self.layer_dropout)

        self.logits = self.projection_layer(feature_vec)
        self.predictions = tf.argmax(self.logits, -1)
        if not predict:
            self.loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits))

            self.mpc, self.per_class = tf.metrics.mean_per_class_accuracy(self.labels, self.predictions,
                                                                          self.args.num_labels)

        if train:
            self.lr = tf.Variable(self.args.learning_rate, trainable=False, name="lr")
            lr_scalar = tf.summary.scalar("lr", self.lr, family="train")

            self.optimizer = tf.train.AdamOptimizer(self.lr)  # tf.train.RMSPropOptimizer(self.lr)
            gradients, variables = zip(*self.optimizer.compute_gradients(self.loss))
            clipped_gradients, self.gradient_norm = tf.clip_by_global_norm(gradients, 3.5)
            self.gradient_scalar = tf.summary.scalar("gradient_norm", self.gradient_norm, family="train")
            self.train_op = self.optimizer.apply_gradients(zip(clipped_gradients, variables),
                                                           global_step=tf.train.get_or_create_global_step())

            self.merge = tf.summary.merge([self.gradient_scalar, lr_scalar])

    @property
    def inpts(self):
        return (
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
        )

    def extract_node_children(self, batch_size, child_indices, feature_tokens, top_rnn_output, child_ids, segment_ids,
                              lengths):
        cids = tf.to_int64(child_indices)
        with tf.control_dependencies([tf.assert_less(tf.reduce_max(child_indices), tf.reduce_max(lengths)),
                                      tf.assert_greater_equal(child_indices, 0)]):
            first_dim = tf.to_int64(segment_ids)
            second_dim = cids
            selector = tf.concat([tf.expand_dims(first_dim, -1), tf.expand_dims(second_dim, 1)], -1)
            rep = tf.gather_nd(top_rnn_output, selector)

            encoding = get_timing_signal_1d(cids, top_rnn_output.shape[-1])
            rep += encoding

            child_types = tf.nn.embedding_lookup(self.edge_embeddings, self.child_edge_types)
            child_resh = self.child_processing_layers(tf.concat((rep, child_types), -1))
            child_resh = tf.RaggedTensor.from_row_lengths(child_resh, tf.to_int64(child_ids))
            reduced = tf.reduce_max(child_resh, axis=1)
            return tf.reshape(reduced, [batch_size, feature_tokens, 512])

    def extract_vectors_3d(self, first_d, second_d, batch_size, n, t, lengths):
        """
        Extracts `n` vectors from the last dimension of `t`.

        :param first_d: a vector that indexes into the first dimension of `t`.
        :param second_d: a matrix with len(first_d) rows and `n` columns that index into the second dimension of `t`
        :param batch_size:
        :param n:
        :param t:
        :return: the vectors indexed by first_d and second_d
        """

        with tf.control_dependencies(
                [tf.assert_less(tf.reduce_max(second_d, axis=1), lengths), tf.assert_greater_equal(second_d, 0)]):
            indices = tf.reshape(second_d, shape=[batch_size, n, 1])
            selectors = tf.concat(
                [tf.tile(
                    tf.expand_dims(first_d, 1),
                    [1, n, 1]
                ), indices],
                -1)
            features = tf.gather_nd(t, selectors)
            return features


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
                accuracy = tf.reduce_mean(
                    tf.to_float(tf.equal(predictions, labels)))
                x_ent = self.loss(logits, labels=labels)

            gradients = tape.gradient(x_ent, self.weights())
            self.optimizer.apply_gradients(zip(gradients, self.weights()),
                                           global_step=tf.train.get_or_create_global_step())

        else:
            logits = self(feats)
            predictions = tf.to_int32(tf.argmax(logits, axis=-1))
            accuracy = tf.reduce_mean(
                tf.to_float(tf.equal(predictions, labels)))
            x_ent = self.loss(logits, labels=labels)

        return sum(x_ent.numpy()), accuracy

    def restore(self, file_prefix, args):
        # Initialize layers with an empty batch
        self(np.zeros(
            (1, args.shapes.max_stack_size +
             args.shapes.max_buffer_size + args.max_training_length + 1),
            dtype=np.int32
        ))
        super().restore(file_prefix)

    def build_encoder(self):
        encoder = []
        for _ in range(self.layers):
            encoder.append(SelfAttention(
                self.args.num_heads, self.args.embedding_size))
            encoder.append(ResidualFeedforward(
                self.args.embedding_size, self.args.self_attention_neurons))

        return encoder
