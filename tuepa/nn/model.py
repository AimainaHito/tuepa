from .base_model import BaseModel
from .blocks import *

import tensorflow as tf
import numpy as np
import json


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
    def __init__(self, args, num_labels, num_dependencies, num_pos, num_ner, train, predict=False, batch=None):
        super().__init__()
        self.args = args
        word_dim = 1024 if not args.features.word.finalfrontier else 1324

        feature_tokens = self.args.shapes.max_buffer_size + self.args.shapes.max_stack_size
        # Inputs
        self.num_feature_tokens = feature_tokens
        if not predict:
            self.create_placeholders(feature_tokens, word_dim)
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
                self.action_counts,
                self.action_ratios,
                self.node_ratios,
                self.root,
            ) = batch

        if not predict:
            self.labels = tf.placeholder(name="labels", shape=[None], dtype=tf.int32)

        # Utility
        batch_size = tf.shape(self.form_indices)[0]
        batch_indices = tf.expand_dims(tf.range(batch_size, dtype=tf.int32), 1)
        conc = lambda x, y: tf.concat([x, y], -1)

        self.init_variables(args, feature_tokens, num_dependencies, num_labels, num_ner, num_pos)

        # prepend padding and non terminal embedding, non-terminals + padded positions on the stack have form index
        # 0 and 1, the rest is offset by 2
        top_rnn_output = tf.concat(
            [tf.tile(self.padding_embedding, [batch_size, 1, 1]),
             tf.tile(self.non_terminal_embedding, [batch_size, 1, 1]), self.elmo],
            1)

        # dense layers
        self.feed_forward_layers = feed_forward_from_json(json.loads(args.model.layers))
        self.child_processing_layers = feed_forward_from_json(json.loads(args.model.child_processing_layers))
        self.projection_layer = tf.layers.Dense(num_labels, use_bias=False)

        # global features
        global_features = self.global_features(batch_size, conc)

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
                                                    segment_ids=self.batch_ind, lengths=self.sentence_lengths + 2,
                                                    train=train)

        # per token features
        per_token_features = self.token_features(batch_size, conc, feature_tokens)

        features = tf.concat(
            [form_features, head_features, child_features, per_token_features],
            axis=-1)

        feedforward_input = tf.reshape(features, [batch_size, features.shape[1] * features.shape[2]])
        feature_vec = tf.concat([global_features, feedforward_input], -1)

        if train:
            feature_vec = tf.nn.dropout(feature_vec, args.training.input_dropout)

        for layer in self.feed_forward_layers:
            if feature_vec.shape[-1] == layer.units:
                feature_vec = layer(feature_vec) + feature_vec
            else:
                feature_vec = layer(feature_vec)
            if train:
                feature_vec = tf.nn.dropout(feature_vec, args.training.layer_dropout)

        self.logits = self.projection_layer(feature_vec)
        self.predictions = tf.argmax(self.logits, -1)

        if not predict:
            self.loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits))

            self.mpc, self.per_class = tf.metrics.mean_per_class_accuracy(self.labels, self.predictions,
                                                                          self.args.num_labels)

        if train:
            self.lr = tf.placeholder_with_default(args.training.learning_rate, [])
            lr_scalar = tf.summary.scalar("lr", self.lr, family="train")
            self.optimizer = tf.train.AdamOptimizer(self.lr)
            gradients, variables = zip(*self.optimizer.compute_gradients(self.loss))
            clipped_gradients, self.gradient_norm = tf.clip_by_global_norm(gradients, 10)
            self.gradient_scalar = tf.summary.scalar("gradient_norm", self.gradient_norm, family="train")
            self.train_op = self.optimizer.apply_gradients(zip(clipped_gradients, variables),
                                                           global_step=tf.train.get_or_create_global_step())

            self.merge = tf.summary.merge([self.gradient_scalar, lr_scalar])

    def create_placeholders(self, feature_tokens, word_dim):
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
        self.action_ratios = tf.placeholder(name="action_ratios", shape=[None], dtype=tf.float32)
        self.node_ratios = tf.placeholder(name="node_ratios", shape=[None], dtype=tf.float32)
        self.action_counts = tf.placeholder(name="act_counts", shape=[None, self.args.num_labels],
                                            dtype=tf.int32)
        self.root = tf.placeholder(name="root", shape=[None, feature_tokens], dtype=tf.int32)
        self.elmo = tf.placeholder(name="elmo", shape=[None, None, word_dim], dtype=tf.float32)

    def init_variables(self, args, feature_tokens, num_dependencies, num_labels, num_ner, num_pos):
        # Embeddings
        if args.features.glob.history:
            self.history_embeddings = tf.get_variable(
                name="history_embeddings",
                shape=[num_labels, args.model.history_embedding_size],
                dtype=tf.float32
            )
        if args.features.token.pos:
            self.pos_embeddings = tf.get_variable(
                name="pos_embeddings",
                shape=[num_pos, args.model.embedding_size],
                dtype=tf.float32
            )
        if args.features.token.dep:
            self.dep_embeddings = tf.get_variable(
                name="dep_embeddings",
                shape=[num_dependencies, args.model.embedding_size],
                dtype=tf.float32
            )
        if args.features.token.ner:
            self.ner_embeddings = tf.get_variable(
                name="ner_embeddings",
                shape=[num_ner, args.model.embedding_size],
                dtype=tf.float32
            )
        if args.features.token.children:
            self.edge_embeddings = tf.get_variable(
                name="edge_embeddings",
                shape=[args.num_edges, args.model.embedding_size],
                dtype=tf.float32)
        if args.features.glob.action_counts:
            self.action_count_embeddings = tf.get_variable(
                name="ac_count",
                shape=[1, num_labels, max(self.args.model.embedding_size // 5, 10)],
                dtype=tf.float32)
        if args.features.token.height:
            self.height_embeddings = tf.get_variable(
                name="height_1",
                shape=[1, feature_tokens, max(self.args.model.embedding_size // 5, 10)],
                dtype=tf.float32)
        if args.features.token.inc:
            self.incoming_embedding = tf.get_variable(
                name="inc_1",
                shape=[1, feature_tokens, args.num_edges, max(self.args.model.embedding_size // 5, 10)],
                dtype=tf.float32
            )
        if args.features.token.out:
            self.out_embedding = tf.get_variable(
                name="out_1",
                shape=[1, feature_tokens, args.num_edges, max(self.args.model.embedding_size // 5, 10)],
                dtype=tf.float32
            )

        self.non_terminal_embedding = tf.get_variable("non_terminal", shape=[1, 1, self.elmo.shape[-1]],
                                                      dtype=tf.float32)
        self.padding_embedding = tf.get_variable("padding", shape=[1, 1, self.elmo.shape[-1]], dtype=tf.float32)

    def token_features(self, batch_size, conc, feature_tokens):
        features = []

        if self.args.features.token.is_root:
            root = tf.expand_dims(tf.to_float(self.root), -1)
            features.append(root)

        if self.args.features.token.height:
            height = tf.tile(self.height_embeddings, [batch_size, 1, 1])
            height = conc(height, get_timing_signal_1d(self.height, self.height_embeddings.shape[-1].value))
            height = tf.reshape(height, [batch_size, feature_tokens, self.incoming_embedding.shape[-1] * 2])
            features.append(height)

        if self.args.features.token.inc:
            inc = tf.tile(self.incoming_embedding, [batch_size, 1, 1, 1])
            inc = conc(inc, get_timing_signal_1d(self.inc, self.incoming_embedding.shape[-1].value))
            inc = tf.reshape(inc,
                             [batch_size, feature_tokens, self.args.num_edges * self.incoming_embedding.shape[-1] * 2])
            features.append(inc)

        if self.args.features.token.out:
            out = tf.tile(self.out_embedding, [batch_size, 1, 1, 1])
            out = conc(out, get_timing_signal_1d(self.inc, self.out_embedding.shape[-1].value))
            out = tf.reshape(out,
                             [batch_size, feature_tokens, self.args.num_edges * self.incoming_embedding.shape[-1] * 2])
            features.append(out)

        if self.args.features.token.pos:
            pos_features = tf.nn.embedding_lookup(self.pos_embeddings, self.pos)
            features.append(pos_features)

        if self.args.features.token.dep:
            dep_features = tf.nn.embedding_lookup(self.dep_embeddings, self.dep_types)
            features.append(dep_features)

        if self.args.features.token.ner:
            ner_features = tf.nn.embedding_lookup(self.ner_embeddings, self.ner)
            features.append(ner_features)

        return tf.concat(features, -1)

    def global_features(self, batch_size, conc):
        features = []
        if self.args.features.glob.action_counts:
            action_counts = tf.tile(self.action_count_embeddings, [batch_size, 1, 1])
            action_counts = conc(action_counts,
                                 get_timing_signal_1d(self.action_counts, self.action_count_embeddings.shape[-1].value))
            action_counts = tf.reshape(action_counts, [batch_size, self.args.num_labels * 10 * 2])
            features.append(action_counts)

        if self.args.features.glob.history:
            history_input = tf.nn.embedding_lookup(self.history_embeddings, self.history)
            history_input = tf.reshape(history_input, [batch_size, 10 * self.history_embeddings.shape[-1]])
            features.append(history_input)

        if self.args.features.glob.action_ratio:
            action_ratios = tf.expand_dims(self.action_ratios, -1)
            features.append(action_ratios)

        if self.args.features.glob.node_ratio:
            node_ratios = tf.expand_dims(self.node_ratios, -1)
            features.append(node_ratios)
        return tf.concat(features, -1)

    @property
    def placeholders(self):
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
            self.action_counts,
            self.action_ratios,
            self.node_ratios,
            self.root,
        )

    def extract_node_children(self, batch_size, child_indices, feature_tokens, top_rnn_output, child_ids, segment_ids,
                              lengths, train):
        child_aggregator = self.args.model.child_aggregator
        with tf.control_dependencies([tf.assert_less(tf.reduce_max(child_indices), tf.reduce_max(lengths)),
                                      tf.assert_greater_equal(child_indices, 0)]):
            first_dim = tf.to_int64(segment_ids)
            second_dim = tf.to_int64(child_indices)
            selector = tf.concat([tf.expand_dims(first_dim, -1), tf.expand_dims(second_dim, 1)], -1)
            rep = tf.gather_nd(top_rnn_output, selector)

            encoding = get_timing_signal_1d(second_dim, top_rnn_output.shape[-1])
            rep += encoding

            child_types = tf.nn.embedding_lookup(self.edge_embeddings, self.child_edge_types)
            rep = tf.concat((rep, child_types), -1)
            for l in self.child_processing_layers:
                if train:
                    rep = tf.nn.dropout(rep, self.args.training.child_dropout)
                rep = l(rep)

            child_resh = tf.RaggedTensor.from_row_lengths(rep, tf.to_int64(child_ids))
            if child_aggregator == 'max':
                reduced = tf.reduce_max(child_resh, axis=1)
            elif child_aggregator == 'mean':
                reduced = tf.reduce_mean(child_resh, axis=1)
            else:
                raise NotImplementedError("Child aggregation method: '{}' does not exist!".format(child_aggregator))

            return tf.reshape(reduced, [batch_size, feature_tokens, self.child_processing_layers[-1].units])

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
