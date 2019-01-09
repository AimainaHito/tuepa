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
from tuepa.data.elmo import preprocess_dataset, specific_elmo
from tuepa.data.preprocessing import read_passages
import tensorflow as tf
import sys
import tuepa.nn.bert.modeling as modeling
import tensorflow as tf

from tensorflow.contrib import autograph


oracle_parser = get_oracle_parser()
constructions.add_argument(oracle_parser)
argument_parser = get_preprocess_parser(parents=[oracle_parser])
argument_parser.add_argument("bert", help="path to bert model.")
argument_parser.add_argument("-b","--batch_size",default=32, help="batch size.")

args = argument_parser.parse_args(sys.argv[1:])

tokenizer = tokenization.FullTokenizer(
    vocab_file=os.path.join(args.bert_model, "/vocab.txt"), do_lower_case=False)

label_numberer = Numberer()
pos_numberer = Numberer(first_elements=["<PAD>"])
dep_numberer = Numberer(first_elements=["<PAD>"])
edge_numberer = Numberer(first_elements=["<PAD>"])

passages = list(read_passages([sys.argv[1]]))


def preprocess(passages, train):
    sents = [[node.text for node in p.layer("0").all] for p in passages]

    tokenized = list(map(lambda x: ["[CLS]"] + [tokenizer.wordpiece_tokenizer.tokenize(t) for t in x] + ["[SEP]"], sents))

    seg_ids = []
    sents = []
    masks = []
    select = []
    targets = []
    go = label_numberer.number("<GO>", train=True)
    stop = label_numberer.number("<STOP>", train=True)

    for p, sent in zip(passages, tokenized):
        targets.append(
            (go,)+label_numberer.number_sequence(str(p), train=train)+(stop,))

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


sents, targets, masks = preprocess(passages=list(
    read_passages([sys.argv[1]])), train=True)
val_sents, val_targets, val_masks = preprocess(
    passages=list(read_passages([sys.argv[2]])), train=False)

np_sents, np_targets, np_lens, np_masks = py2numpy(sents, targets, masks)
val_np_sents, val_np_targets, val_np_lens, val_np_masks = p2numpy(
    val_sents, val_targets, val_masks)

@autograph.convert(recursive=True, verbose=autograph.Verbosity.VERBOSE)
def decode(output, input_embeddings, projection_layer, train, label_numberer, out_vocab, cell):
    predictions = tf.TensorArray(tf.float32, 0, True)
    autograph.set_element_type(predictions, tf.float32)
    i = tf.constant(0)
    state = tf.layers.dense(output[:, 0], 300)
    if train:
        while i < tf.shape(input_embeddings)[1]:
            # print(state.shape)
            # print(input_embeddings[:,i])
            _, state = cell(input_embeddings[:, i], state)
            prediction = projection_layer(state)
            predictions.append(prediction)
            # mask losses
            i += 1
    else:
        next_input = tf.nn.embedding_lookup(params=out_vocab, ids=tf.tile(
            [label_numberer.value2num["<GO>"]], [tf.shape(output)[0]]))
        should_stop = False
        while not should_stop and i < 350:
            _, state = cell(next_input, state)
            prediction = projection_layer(state)
            next_input = tf.nn.embedding_lookup(
                params=out_vocab, ids=tf.argmax(prediction, -1))
            p = tf.argmax(prediction, -1)
            should_stop = tf.reduce_all(
                p == label_numberer.value2num["<STOP>"])
            predictions.append(prediction)
            i += 1
    logits = tf.transpose(autograph.stack(predictions), [1, 0, 2])
    return logits


def create_graph():
    bert_config = modeling.BertConfig.from_json_file(
        os.path.join(args.bert_model, "bert_config.json"))
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

        output_vocabulary = tf.get_variable(
            "out_voc", shape=[label_numberer.max, 128], dtype=tf.float32)
        projection_layer = tf.layers.Dense(label_numberer.max, use_bias=False)
        input_embeddings = tf.nn.embedding_lookup(
            output_vocabulary, target[:, :-1])

        model = modeling.BertModel(
            config=bert_config,
            is_training=True,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=False)
        bert_out = model.sequence_output
        cell = tf.nn.rnn_cell.GRUCell(300, name="gu")
        train_logits = decode(bert_out, input_embeddings, projection_layer,
                              train=True, label_numberer=label_numberer, out_vocab=output_vocabulary, cell=cell)
        train_logits = tf.pad(train_logits,
                              [[0, 0], [0, tf.maximum((tf.shape(target)[1]-1) - tf.shape(train_logits)[1], 0)],
                               [0, 0]])
        train_x_ent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=target[:, 1:],
            logits=train_logits)
        mask = tf.sequence_mask(target_lens, tf.shape(
            train_logits)[1], dtype=tf.float32)
        train_loss = train_x_ent*mask

        train_op = tf.train.AdamOptimizer().minimize(tf.reduce_mean(train_loss))

        val_logits = decode(bert_out, None, projection_layer, train=False,
                            label_numberer=label_numberer, out_vocab=output_vocabulary, cell=cell)
    return g, train_loss, train_logits, train_op, input_ids, target, target_lens, input_mask, val_logits


g, train_loss, train_logits, train_op, input_ids, target, target_lens, input_mask, val_logits = create_graph()


gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(gpu_options=gpu_options)

with tf.Session(config=config, graph=g) as sess:
    tvars = tf.trainable_variables()
    sess.run(tf.global_variables_initializer())
    (assignment_map, initialized_variable_names
     ) = modeling.get_assignment_map_from_checkpoint(tvars, os.path.join(args.bert_model, "bert_model.ckpt"))
    tf.train.init_from_checkpoint(os.path.join(args.bert_model, "bert_model.ckpt"), assignment_map)
    while True:
        for n in range(len(np_sents)//8):
            sl = slice(n*8, min((n+1)*8, len(np_sents)))
            s = np_sents[sl]
            t = np_targets[sl]
            print(t.shape)
            l = np_lens[sl]
            m = np_masks[sl]
            lo, logs, _ = sess.run([train_loss, train_logits, train_op], feed_dict={
                input_ids: s,
                target: t,
                target_lens: l,
                input_mask: m,
            })
            print(lo.sum())
            preds = logs.argmax(-1)
            for p in preds:
                print(label_numberer.decode_sequence(p, "<STOP>"))
            print(np.equal(preds, t[:, 1:]).mean())
        for n in range(len(val_np_sents//8)):
            sl = slice(n*8, min((n+1)*8, len(val_np_sents)))
            s = val_np_sents[sl]
            t = val_np_targets[sl]
            print(t.shape)
            l = val_np_lens[sl]
            m = val_np_masks[sl]
            logs = sess.run(val_logits, feed_dict={
                input_ids: s,
                target: t,
                target_lens: l,
                input_mask: m,
            })
            preds = logs.argmax(-1)
            for p in preds:
                print(label_numberer.decode_sequence(p, "<STOP>"))
            # print(np.equal(preds, t[:, 1:]).mean())
