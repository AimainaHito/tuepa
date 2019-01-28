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


class ElModel(BaseModel):
    def __init__(self, args, num_labels, num_dependencies, num_pos, num_ner):
        super().__init__()
        self.args = args

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
        
        self.edge_embeddings = tf.get_variable(name="edge_embeddings",shape=[args.num_edges,args.embedding_size], dtype=tf.float32)

        if self.args.numbers == "embed":
            self.number_embeddings = tf.get_variable(
                name="numbers",
                shape=[args.num_num, max(self.args.embedding_size // 5, 10)],
                dtype=tf.float32)
        

        self.softmax_w = tf.get_variable("elmo_weight", dtype=tf.float32,trainable=True, initializer=[[[1.],[1.],[1.]]])
        self.elmo_scale = tf.get_variable("elmo_scale",initializer=1., dtype=tf.float32,trainable=True)

        self.non_terminal_embedding = tf.get_variable("non_terminal", shape=[1, 1, 1024],
                                                      dtype=tf.float32)
        self.padding_embedding = tf.get_variable(
            "padding", shape=[1, 1, 1024], dtype=tf.float32)
        self.cls = tf.get_variable("cls", shape=[1, 1, 1024], dtype=tf.float32)
        self.self_attention_encoder = onmt.encoders.SelfAttentionEncoder(3, 1024)
        # history rnn
        self.history_rnn = tf.layers.Dense(args.history_rnn_neurons,activation=tf.nn.relu)#tf.keras.layers.LSTM(args.history_rnn_neurons, return_sequences=True)

        # dense layers
        self.feed_forward_layers = feed_forward_from_json(json.loads(args.layers))

        self.downsampling_layer = tf.layers.Dense(
            self.feed_forward_layers[0].input_size if isinstance(self.feed_forward_layers[0], UpDownWithResiduals) else
            self.feed_forward_layers[0].units,use_bias=False)

        self.child_processing_layers = [tf.layers.Dense(self.args.top_rnn_neurons,activation=tf.nn.relu)]

        self.projection_layer = tf.layers.Dense(
            num_labels, use_bias=False, activation=None)


        self.lr = tf.Variable(args.learning_rate, name="learning_rate")
        self.optimizer = tf.train.AdamOptimizer(args.learning_rate)

        self.input_dropout = args.input_dropout
        self.layer_dropout = args.layer_dropout
        self.__call__ = tf.contrib.eager.defun(self.__call__)
        self._create_inputs()

    def _create_inputs(self):
        feature_tokens = self.args.shapes.max_buffer_size + self.args.shapes.max_stack_size
        self.num_feature_tokens = feature_tokens
        self.form_indices = tf.placeholder(name="form_indices", shape=[None, feature_tokens], dtype=tf.int32)
        self.dep_types = tf.placeholder(name="dep_types", shape=[None, feature_tokens], dtype=tf.int32)
        self.head_indices = tf.placeholder(name="head_indices", shape=[None, feature_tokens], dtype=tf.int32)
        self.pos = tf.placeholder(name="pos", shape=[None, feature_tokens], dtype=tf.int32)
        self.child_indices = tf.placeholder(name="head_indices", shape=[None, feature_tokens,60], dtype=tf.int32)
        self.child_edge_types = tf.placeholder(name="head_indices", shape=[None, feature_tokens,60], dtype=tf.int32)
        self.ner = tf.placeholder(name="ner", shape=[None, feature_tokens], dtype=tf.int32)
        self.height = tf.placeholder(name="height", shape=[None, feature_tokens], dtype=tf.int32)
        self.inc = tf.placeholder(name="inc", shape=[None, feature_tokens, self.args.num_edges], dtype=tf.int32)
        self.out = tf.placeholder(name="out", shape=[None, feature_tokens, self.args.num_edges], dtype=tf.int32)
        self.history = tf.placeholder(name="hist", shape=[None, None], dtype=tf.int32)
        self.elmo = tf.placeholder(name="elmo", shape=[None, None, 1024], dtype=tf.float32)
        self.sentence_lengths = tf.placeholder(name="sent_lens", shape=[None], dtype=tf.int32)
        self.history_lengths = tf.placeholder(name="hist_lens", shape=[None], dtype=tf.int32)
        self.action_ratios = tf.placeholder(name="action_ratios", shape=[None], dtype=tf.float32)
        self.node_ratios = tf.placeholder(name="node_ratios", shape=[None], dtype=tf.float32)
        self.action_counts = tf.placeholder(name="act_counts", shape=[None, self.args.num_labels], dtype=tf.int32)
        self.root = tf.placeholder(name="root", shape=[None, feature_tokens], dtype=tf.int32)


    @property
    def inpts(self):
        return (
            self.form_indices,
            self.dep_types,
            self.head_indices,
            self.pos,
            self.child_indices,
            self.child_edge_types,
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

    def __call__(self, batch, mode=None, train=False, eval=False):
        feature_tokens = self.args.shapes.max_buffer_size + self.args.shapes.max_stack_size
        if eval:
            batch = self.inpts
        form_indices, dep_types, head_indices, pos, child_indices, child_edge_types, ner, height, inc, out, history, elmo, sentence_lengths, history_lengths, action_counts, action_ratios, node_ratios, root = batch
        batch_size = tf.shape(form_indices)[0]
        batch_indices = tf.expand_dims(tf.range(batch_size, dtype=tf.int32), 1)
        self.child_edge_typess = child_edge_types
        if self.args.numbers == 'embed':
            action_counts = tf.reshape(tf.nn.embedding_lookup(self.number_embeddings, action_counts), shape=[
                batch_size, self.args.num_labels * self.number_embeddings.shape[1]])
            inc = tf.reshape(tf.nn.embedding_lookup(self.number_embeddings, inc), shape=[
                batch_size, feature_tokens * self.args.num_edges * self.number_embeddings.shape[1]])
            out = tf.reshape(tf.nn.embedding_lookup(self.number_embeddings, out), shape=[
                batch_size, feature_tokens * self.args.num_edges * self.number_embeddings.shape[1]])
            height = tf.reshape(tf.nn.embedding_lookup(self.number_embeddings, height),
                                [batch_size, feature_tokens * self.number_embeddings.shape[1]])
        elif self.args.numbers == 'absolute':
            action_counts = tf.to_float(tf.reshape(action_counts, shape=[batch_size, self.args.num_labels]))
            inc = tf.to_float(tf.reshape(inc, shape=[batch_size, feature_tokens * self.args.num_edges]))
            out = tf.to_float(tf.reshape(out, shape=[batch_size, feature_tokens * self.args.num_edges]))
            height = tf.to_float(tf.reshape(height, [batch_size, feature_tokens]))
        elif self.args.numbers == 'log':
            action_counts = tf.log(
                tf.to_float(tf.reshape(action_counts, shape=[batch_size, self.args.num_labels])) + 0.0001)
            inc = tf.log(
                tf.to_float(tf.reshape(inc, shape=[batch_size, feature_tokens * self.args.num_edges])) + 0.0001)
            out = tf.log(
                tf.to_float(tf.reshape(out, shape=[batch_size, feature_tokens * self.args.num_edges])) + 0.0001)
            height = tf.log(tf.to_float(tf.reshape(height, [batch_size, feature_tokens])) + 0.0001)

        elmo = elmo * self.elmo_scale
        #top_rnn_output, top_rnn_state,means = self.encode_elmo(batch_indices, elmo, sentence_lengths, train)
        history_rnn_state = self.apply_history_rnn(batch_indices, history, tf.maximum(history_lengths - 1, 0))

        # prepend padding and non terminal embedding, non-terminals + padded positions on the stack have form index
        # 0 and 1, the rest is offset by 2
        top_rnn_output = tf.concat(
            [tf.tile(self.padding_embedding, [batch_size, 1, 1]),
             tf.tile(self.non_terminal_embedding, [batch_size, 1, 1]), elmo],
            1)

        # extract embeddings for stack + buffer tokens
        form_features = self.extract_vectors_3d(first_d=batch_indices,
                                                second_d=form_indices,
                                                batch_size=batch_size,
                                                n=feature_tokens, t=top_rnn_output)

        head_features = self.extract_vectors_3d(first_d=batch_indices,
                second_d=head_indices,
                                                batch_size=batch_size,
                                                n=feature_tokens, t=top_rnn_output)
        pos_features = tf.nn.embedding_lookup(self.pos_embeddings, pos)
        dep_features = tf.nn.embedding_lookup(self.dep_embeddings, dep_types)
        ner_features = tf.nn.embedding_lookup(self.ner_embeddings, ner)
        
        features = tf.concat(
            [form_features, pos_features, dep_features, ner_features], axis=-1)
        feedforward_input = tf.reshape(
            features,
            [batch_size, features.shape[1] * features.shape[2]])

        feature_vec = tf.concat([history_rnn_state, feedforward_input,tf.reshape(head_features,[batch_size,feature_tokens*head_features.shape[-1]]),
                                 height, inc, out, action_counts, tf.expand_dims(action_ratios, -1),
                                 tf.expand_dims(node_ratios, -1), tf.to_float(root)], -1)

        rep = self.extract_node_children(batch_indices, batch_size, child_indices, feature_tokens, top_rnn_output, train=train)

        feature_vec = tf.concat([feature_vec, rep], -1)

        if train:
            feature_vec = tf.nn.dropout(feature_vec, self.input_dropout)

        for layer in self.feed_forward_layers:
            feature_vec = layer(feature_vec)
            if train:
                feature_vec = tf.nn.dropout(feature_vec, self.layer_dropout)
        print(train)
#        feature_vec = self.bn(feature_vec,training=train)
        return self.projection_layer(feature_vec)

    def extract_node_children(self, batch_indices, batch_size, child_indices, feature_tokens, top_rnn_output, train=False):
        with tf.control_dependencies([tf.assert_less(child_indices, tf.shape(top_rnn_output)[1]),tf.assert_greater_equal(child_indices, 0)]):
            feature_tokens = 6#
            child_indices=tf.concat((child_indices[:,slice(0,15,5),:30],child_indices[:,15:,:30]),1)
            first_dim = tf.tile(batch_indices, [1, feature_tokens * child_indices.shape[-1]])
            first_dim = tf.reshape(first_dim, [batch_size * feature_tokens * child_indices.shape[-1], 1])
            second_dim = tf.reshape(child_indices, (batch_size * feature_tokens * child_indices.shape[-1],))
            selector = tf.concat([first_dim, tf.expand_dims(second_dim, 1)], -1)
            node_rep = tf.gather_nd(top_rnn_output, selector)
            rep = tf.reshape(node_rep, [batch_size * feature_tokens, child_indices.shape[2], node_rep.shape[-1]])

            child_types = tf.nn.embedding_lookup(self.edge_embeddings,tf.concat((self.child_edge_typess[:,slice(0,15,5),:30],self.child_edge_typess[:,15:,:30]),axis=1))
            child_resh = tf.reshape(child_types,[tf.shape(batch_indices)[0]*feature_tokens,30,self.args.embedding_size])
            child_resh = tf.concat((rep,child_resh),-1)
            rep = child_resh
            for l in self.child_processing_layers:
                rep = l(rep)
            print(rep)
            rep = tf.reshape(tf.reduce_max(rep,axis=1),(batch_size,feature_tokens*self.args.top_rnn_neurons))
#            rep = tf.reshape(rep,[batch_size,rep.shape[-1]])
            return rep

    def apply_history_rnn(self, batch_indices, history, history_lengths):
        history_input = tf.nn.embedding_lookup(
                self.history_embeddings, tf.reshape(history[:,-10:],[tf.shape(batch_indices)[0],10]))
        history_rnn_outputs = self.history_rnn(tf.reshape(history_input, [tf.shape(batch_indices)[0],10*self.history_embeddings.shape[-1]]))
        return history_rnn_outputs

    def extract_vectors_3d(self, first_d, second_d, batch_size, n, t):
        """
        Extracts `n` vectors from the last dimension of `t`.

        :param first_d: a vector that indexes into the first dimension of `t`.
        :param second_d: a matrix with len(first_d) rows and `n` columns that index into the second dimension of `t`
        :param batch_size:
        :param n:
        :param t:
        :return: the vectors indexed by first_d and second_d
        """

        with tf.control_dependencies([tf.assert_less(second_d, tf.shape(t)[1]),tf.assert_greater_equal(second_d, 0)]):
            indices = tf.reshape(second_d, shape=[batch_size, n, 1])
            selectors = tf.concat(
                [tf.tile(
                    tf.expand_dims(first_d, 1),
                    [1, n, 1]
                ), indices],
                -1)
            features = tf.gather_nd(t, selectors)
            return features

    def encode_elmo(self, batch_indices, elmo, sentence_lengths,train=False):
        """
        Runs bi-rnn over `elmo` and extracts the final states at position `sentence_lengths`
        :param batch_indices: range vector with length of batch size
        :param elmo: tensor containing elmo output for the input sentences
        :param sentence_lengths: length of elmo sentences
        :return: final states of the bi rnn over the elmo sentences
        """
        elmo = tf.concat([tf.tile(self.cls, [tf.shape(batch_indices)[0], 1, 1]), elmo], 1)

        with tf.variable_scope("elmo_self_attention",reuse=False): 
            (outputs, state, sequence_length) = self.self_attention_encoder.encode(tf.convert_to_tensor(elmo),
                                                                               sentence_lengths + 1, tf.estimator.ModeKeys.TRAIN if train else tf.estimator.ModeKeys.PREDICT)
        return outputs[:,1:], outputs[:,0], state

    def compute_gradients(self, batch, labels):
        logits = self(batch, train=True)
        predictions = tf.to_int32(tf.argmax(logits, axis=-1))
        accuracy = tf.reduce_mean(tf.to_float(tf.equal(predictions, labels)))
        return accuracy, self.loss(logits, labels=labels)


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
