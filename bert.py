from __future__ import print_function
import numpy as np
import IPython
from tuepa.data.elmo import *
import tuepa.nn.bert.optimization as optimization
import tuepa.nn.bert.tokenization as tokenization
import sys
import os
import sys
from ucca import constructions
from tuepa.util.config import get_preprocess_parser, get_oracle_parser, save_args, load_args, ARGS_FILENAME, LABELS_FILENAME, \
    DEP_FILENAME, EDGE_FILENAME, POS_FILENAME
from tuepa.util.numberer import Numberer
from tuepa.data.preprocessing import read_passages
import sys
import tuepa.nn.bert.modeling as modeling
import tensorflow as tf

from tensorflow.contrib import autograph


def preprocess(passages, train):
    sents = [[node.text for node in p.layer("0").all] for p in passages]

    tokenized = list(map(lambda x: [
                     "[CLS]"] + [tokenizer.wordpiece_tokenizer.tokenize(t) for t in x] + ["[SEP]"], sents))

    sents = []
    masks = []
    select = []
    targets = []
    go = label_numberer.number("<GO>", train=True)
    stop = label_numberer.number("<STOP>", train=True)

    for p, sent in zip(passages, tokenized):
        t = (go,)+label_numberer.number_sequence(str(p), train=train)+(stop,)
        if len(t) > 500:
            continue
        targets.append(t)

        sent_ids = []
        total_index = 1  # offset 1 by [CLS]
        sent_selectors = []
        mult2word = []
        for token_index, multi_token in enumerate(sent):
            if len(multi_token) == 1:
                mult2word.append(token_index)
                sent_ids.append(tokenizer.vocab[multi_token[0]])
                sent_selectors.append([total_index])
                total_index += 1
            else:
                selectors = []
                for piece in multi_token:
                    mult2word.append(token_index)
                    sent_ids.append(tokenizer.vocab[piece])
                    selectors.append(total_index)
                    total_index += 1
                sent_selectors.append(selectors)
        masks.append([1]*len(sent_ids))
        sents.append(sent_ids)
        select.append(sent_selectors)
    return sents, targets, masks


def py2numpy(sents, targets, masks):
    np_sents = np.zeros(
        shape=(len(sents), max(map(len, sents))), dtype=np.float32)
    np_targets = np.zeros(
        shape=(len(sents), max(map(len, targets))), dtype=np.float32)
    np_lens = np.zeros(shape=(len(sents),), dtype=np.int32)
    np_masks = np.zeros_like(np_sents)
    n = 0
    for s, t, m in zip(sents, targets, masks):
        print(len(s), len(t))
        np_sents[n, :len(s)] = s
        np_targets[n, :len(t)] = t
        np_masks[n, :len(s)] = m
        np_lens[n] = len(t)
        n += 1
    return np_sents, np_targets, np_lens, np_masks




@autograph.convert(recursive=True, verbose=autograph.Verbosity.VERBOSE)
def decode(state,outputs, input_embeddings, projection_layer, train, label_numberer, out_vocab, cell, V, W2):
    predictions = tf.TensorArray(tf.float32, 0, True)
    autograph.set_element_type(predictions, tf.float32)
    i = tf.constant(0)
    batch_size = tf.shape(outputs)[0]

    if train:
        while i < tf.shape(input_embeddings)[1]:
            # print(state.shape)
            # print(input_embeddings[:,i])
            q = tf.reshape(state.h,[batch_size,1,300])
            # score shape == (batch_size, max_length, 1)
            # we get 1 at the last axis because we are applying tanh(FC(EO) + FC(H)) to self.V
            score = V(tf.nn.tanh(outputs + W2(q)))

            # attention_weights shape == (batch_size, max_length, 1)
            attention_weights = tf.nn.softmax(score, axis=1)
            print(attention_weights[0])
            # context_vector shape after sum == (batch_size, hidden_size)
            context_vector = attention_weights * outputs
            context_vector = tf.reduce_sum(context_vector, axis=1)

            # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
            x = tf.concat([context_vector, input_embeddings[:,i]], axis=-1)
            _, state = cell(x, state)
            prediction = projection_layer(state.h)
            predictions.append(prediction)
            # mask losses
            i += 1
    else:
        next_input = tf.nn.embedding_lookup(params=out_vocab, ids=tf.tile(
            [1], [tf.shape(outputs)[0]]))
        should_stop = False
        while not should_stop and i < 500:
            q = tf.reshape(state.h, [batch_size, 1, 300])
            score = V(tf.nn.tanh(outputs + W2(q)))

            # attention_weights shape == (batch_size, max_length, 1)
            attention_weights = tf.nn.softmax(score, axis=1)
            print(attention_weights[0])
            # context_vector shape after sum == (batch_size, hidden_size)
            context_vector = attention_weights * outputs
            context_vector = tf.reduce_sum(context_vector, axis=1)

            # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
            next_input = tf.concat([context_vector, next_input], axis=-1)

            _, state = cell(next_input, state)

            prediction = projection_layer(state.h)
            next_input = tf.nn.embedding_lookup(
                params=out_vocab, ids=tf.argmax(prediction, -1))
            p = tf.argmax(prediction, -1)
            should_stop = tf.reduce_all(
                p == 2)
            predictions.append(prediction)
            i += 1
    logits = tf.transpose(autograph.stack(predictions), [1, 0, 2])
    return logits


def create_graph(label_numberer):
    bert_config = modeling.BertConfig.from_json_file(
        os.path.join(args.bert, "bert_config.json"))
    g = tf.Graph()
    with g.as_default():
        input_ids = tf.placeholder(
            name="tids", shape=[None, None], dtype=tf.int32)
        segment_ids = input_mask = tf.placeholder(
            name="in_mask", shape=[None, None], dtype=tf.int32)
        # multi_tokens = tf.placeholder(
        #     name="m_toikens", shape=[None, None], dtype=tf.int32)
        target = tf.placeholder(name="target", shape=[
                                None, None], dtype=tf.int32)
        target_lens = tf.placeholder(
            name="target_lens", shape=[None], dtype=tf.int32)
        train = tf.placeholder(name="train",shape=[],dtype=tf.bool)
        output_vocabulary = tf.get_variable(
            "out_voc", shape=[label_numberer.max, 128], dtype=tf.float32)
        projection_layer = tf.layers.Dense(label_numberer.max, use_bias=False)
        input_embeddings = tf.nn.embedding_lookup(
            output_vocabulary, target[:,:-1])

        model = modeling.BertModel(
            config=bert_config,
            is_training=True,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=False)
        bert_out = model.sequence_output
        cell = tf.nn.rnn_cell.LSTMCell(300, name="gu")
        outputs = tf.layers.dense(tf.reshape(bert_out, [tf.shape(bert_out)[0], -1, 768]), 300)
        state_c = tf.layers.dense(model.get_pooled_output(), 300, name="state_reducer")
        state_h = tf.layers.dense(model.get_pooled_output(), 300, name="state_reducer_h")
        state = tf.nn.rnn_cell.LSTMStateTuple(c=state_c,h=state_h)
        V = tf.layers.Dense(1)
        W2 = tf.layers.Dense(300)
        logits = decode(state,outputs, input_embeddings, projection_layer,train=train, label_numberer=label_numberer, out_vocab=output_vocabulary, cell=cell,V=V, W2=W2)
        train_logits = tf.pad(logits,
                              [[0, 0], [0, tf.maximum((tf.shape(target)[1]-1) - tf.shape(logits)[1], 0)],
                               [0, 0]])
        train_x_ent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=target[:, 1:],
            logits=train_logits)
        mask = tf.sequence_mask(target_lens, tf.shape(
            train_logits)[1], dtype=tf.float32)
        train_loss = train_x_ent*mask

        train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(tf.reduce_mean(train_loss,axis=0))

    return g, train_loss, logits, train_op, input_ids, target, target_lens, input_mask, train

if __name__ == "__main__":
    oracle_parser = get_oracle_parser()
    constructions.add_argument(oracle_parser)
    argument_parser = get_preprocess_parser(parents=[oracle_parser])
    argument_parser.add_argument("bert", help="path to bert model.")
    argument_parser.add_argument("-b", "--batch_size", default=16, help="batch size.")

    args = argument_parser.parse_args(sys.argv[1:])

    tokenizer = tokenization.FullTokenizer(
        vocab_file=os.path.join(args.bert, "vocab.txt"), do_lower_case=False)

    label_numberer = Numberer(first_elements=["<PAD>","<GO>","<STOP>"])
    pos_numberer = Numberer(first_elements=["<PAD>"])
    dep_numberer = Numberer(first_elements=["<PAD>"])
    edge_numberer = Numberer(first_elements=["<PAD>"])

    passages = list(read_passages([sys.argv[1]]))


    sents, targets, masks = preprocess(passages=list(
        read_passages([args.training_path])), train=True)
    val_sents, val_targets, val_masks = preprocess(
        passages=list(read_passages([args.validation_path])), train=False)

    np_sents, np_targets, np_lens, np_masks = py2numpy(sents, targets, masks)
    val_np_sents, val_np_targets, val_np_lens, val_np_masks = py2numpy(
        val_sents, val_targets, val_masks)

    g, train_loss, logits, train_op, input_ids, target, target_lens, input_mask,train = create_graph(label_numberer)


    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options)
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

    with tf.Session(config=config, graph=g) as sess:
        tvars = tf.trainable_variables()
        sess.run(tf.global_variables_initializer())
        (assignment_map, initialized_variable_names
         ) = modeling.get_assignment_map_from_checkpoint(tvars, os.path.join(args.bert, "bert_model.ckpt"))
        tf.train.init_from_checkpoint(os.path.join(
            args.bert, "bert_model.ckpt"), assignment_map)
        while True:
            for n in range(len(np_sents)//16):
                sl = slice(n*16, min((n+1)*16, len(np_sents)))
                if n*16- min((n+1)*16, len(val_np_sents)) >= 0:
                    break
                s = np_sents[sl]
                t = np_targets[sl]
                l = np_lens[sl]
                m = np_masks[sl]
                lo, logs, _ = sess.run([train_loss, logits, train_op], feed_dict={
                    input_ids: s,
                    target: t,
                    target_lens: l,
                    input_mask: m,
                    train: True
                })
                print(lo.sum())
                preds = logs.argmax(-1)
                # for p in preds:
                #     print(label_numberer.decode_sequence(p, "<STOP>"))
                print(np.equal(preds, t[:, 1:]).mean())
            for n in range(len(val_np_sents//16)):
                sl = slice(n*16, min((n+1)*16, len(val_np_sents)))
                if n*16- min((n+1)*16, len(val_np_sents)) >= 0:
                    break
                s = val_np_sents[sl]
                t = val_np_targets[sl]
                l = val_np_lens[sl]
                m = val_np_masks[sl]
                logs = sess.run(logits, feed_dict={
                    input_ids: s,
                    target: t,
                    target_lens: l,
                    input_mask: m,
                    train: False
                })
                preds = logs.argmax(-1)
                for p in preds:
                    print(label_numberer.decode_sequence(p, "<STOP>"))
                # print(np.equal(preds, t[:, 1:]).mean())
