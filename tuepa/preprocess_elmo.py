import os
import sys

import numpy as np
import torch
from elmoformanylangs import Embedder

from config import get_preprocess_parser, get_oracle_parser, save_args, load_args, ARGS_FILENAME, LABELS_FILENAME, \
    DEP_FILENAME, EDGE_FILENAME, POS_FILENAME
from numberer import Numberer, load_numberer_from_file
from preprocessing import read_passages,Shapes

from oracle import Oracle
from states.state import State

def extract_elmo_features(state, label_numberer, dep_numberer, pos_numberer, edge_numberer, train=True):
    stack_features = []
    buffer_features = []
    stack = state.stack
    buffer = state.buffer

    def null_features():
        return [0, 0, 0, 0, [], [], 0]

    def extract_feature(node):
        if node.text is not None:
            form = node.index + 1
            dep = node.extra['dep']
            head = node.extra['head'] + form
            pos = node.extra['tag']
        else:
            form = 1
            dep = "<NT>"
            head = 1
            pos = "<NT>"
        incoming = [edge_numberer.number(e.tag, train) for e in node.incoming]
        outgoing = [edge_numberer.number(e.tag, train) for e in node.outgoing]

        height = node.height
        return [form, dep_numberer.number(dep, train=train), head, pos_numberer.number(pos, train=train), incoming,
                outgoing, height]

    for n in range(5):
        try:
            node = stack[-n]
            stack_features.append(extract_feature(node))

            def try_or_value(val, list, ind):
                try:
                    return list[ind]
                except:
                    return val

            def try_or_function(test_fn, fn, elsefn, node):
                try:
                    test_fn(node)
                    return fn()
                except:
                    return elsefn(node)

            test_fn = lambda x: x != 0
            left_parent = try_or_value(0, node.parents, 0)
            left_parent = try_or_function(test_fn, null_features, extract_feature, left_parent)
            stack_features.append(left_parent)

            right_parent = try_or_value(0, node.parents, -1)
            right_parent = try_or_function(test_fn, null_features, extract_feature, right_parent)
            stack_features.append(right_parent)

            left_child = try_or_value(0, node.children, 0)
            left_child = try_or_function(test_fn, null_features, extract_feature, left_child)
            stack_features.append(left_child)

            right_child = try_or_value(0, node.children, -1)
            right_child = try_or_function(test_fn, null_features, extract_feature, right_child)
            stack_features.append(right_child)

        except IndexError:
            stack_features.append(null_features())
            stack_features.append(null_features())
            stack_features.append(null_features())
            stack_features.append(null_features())
            stack_features.append(null_features())
    for n in range(5):
        try:
            buffer_features.append(extract_feature(buffer[-n]))
        except IndexError:
            buffer_features.append(null_features())

    # [node.text if node.text is not None else "<NT>" for node in state.stack[::-1]]
    # buffer_features = [extract_stackfeature(node) for node in state.buffer]
    history_features = [label_numberer.number(str(action), train=train) for action in
                        state.actions[-min(len(state.actions), 100):]]
    return stack_features, buffer_features, history_features

def preprocess_dataset(path,
                       args,
                       shapes=None,
                       max_features=None,
                       label_numberer=None,
                       pos_numberer=None,
                       dep_numberer=None,
                       edge_numberer=None,
                       train=False):

    if not train:
        max_stack_size = shapes.max_stack_size
        max_buffer_size = shapes.max_buffer_size

    else:
        max_stack_size = max_buffer_size = -1

    stack_and_buffer_features = []
    previous_action_counts = []
    labels = []
    history_features = []
    sentence_lengths = []
    history_lengths = []
    passage_id2sent = []
    state2passage_id = []
    passage_id = 0

    for passage in read_passages([path]):
        sentence = [str(n) for n in passage.layer("0").all]
        if len(sentence) > 100:
            continue

        passage_id2sent.append(sentence)
        sentence_lengths.append(len(sentence))

        state = State(passage, args)
        oracle = Oracle(passage, args)
        state_no = 0
        while not state.finished:
            passage_actions = []
            state2passage_id.append(passage_id)
            actions = oracle.generate_actions(state=state)
            action = next(actions)

            stack_features, buffer_features, state_history = extract_elmo_features(
                state,
                label_numberer,
                pos_numberer=pos_numberer,
                dep_numberer=dep_numberer,
                edge_numberer=edge_numberer,
                train=train
            )

            history_lengths.append(len(state_history))

            label = label_numberer.number(str(action), train=shapes is None)
            passage_actions.append(label)
            state.transition(action)
            action.apply()

            stack_and_buffer_features.append((stack_features, buffer_features))
            history_features.append(state_history)

            if shapes is None:
                max_stack_size = max(len(stack_features), max_stack_size)
                max_buffer_size = max(len(buffer_features), max_buffer_size)

            previous_action_counts.append(passage_actions[:state_no])
            labels.append(label)
            state_no += 1

        if max_features is not None and len(stack_and_buffer_features) >= max_features:
            break

        passage_id += 1

    return stack_and_buffer_features, Shapes(max_stack_size, max_buffer_size), history_features, history_lengths, state2passage_id, passage_id2sent, sentence_lengths, previous_action_counts, labels


def specific_elmo(features, embedder, args, train, write_chunk=8192):
    stack_and_buffer_features, shapes, history_features, history_lengths, state2passage_id, passage_id2sent, sentence_lengths, previous_action_counts, labels = features
    max_stack_size = shapes.max_stack_size
    max_buffer_size = shapes.max_buffer_size
    max_hist_size = np.max(history_lengths)
    num_examples = len(stack_and_buffer_features)

    import h5py_cache
    with h5py_cache.File(args.training_out if train else args.validation_out, 'w',
                         chunk_cache_mem_size=128 * 1024 ** 2) as f:
        s_b = f.create_group("stack_buffer")
        form_matrix = s_b.create_dataset('form_indices',
                                         shape=(num_examples, max_stack_size + max_buffer_size), dtype=np.int32,
                                         fillvalue=0)
        dep_matrix = s_b.create_dataset('dependencies', shape=(num_examples, max_stack_size + max_buffer_size),
                                        dtype=np.int32, fillvalue=0)
        head_matrix = s_b.create_dataset('head_indices', shape=(num_examples, max_stack_size + max_buffer_size),
                                         dtype=np.int32, fillvalue=0)
        height_matrix = s_b.create_dataset('height', shape=(num_examples, max_stack_size + max_buffer_size),
                                           dtype=np.int32, fillvalue=0)
        pos_matrix = s_b.create_dataset('pos', shape=(num_examples, max_stack_size + max_buffer_size),
                                        dtype=np.int32, fillvalue=0)
        out_matrix = s_b.create_dataset('out', shape=(
            num_examples, max_stack_size + max_buffer_size, args.num_edges),
                                        dtype=np.int32, fillvalue=0)
        inc_matrix = s_b.create_dataset('in', shape=(
            num_examples, max_stack_size + max_buffer_size, args.num_edges),
                                        dtype=np.int32, fillvalue=0)
        action_counts = f.create_dataset('action_counts',shape=(num_examples,args.num_labels),dtype=np.int32)
        f.create_dataset('labels', data=np.array(labels))
        history_matrix = f.create_dataset('history_features', shape=(num_examples, max_hist_size),
                                          dtype=np.int32, fillvalue=0)

        form_chunk = np.zeros(shape=(write_chunk, max_stack_size + max_buffer_size), dtype=np.int32)
        dep_chunk = np.zeros(shape=(write_chunk, max_stack_size + max_buffer_size), dtype=np.int32)
        head_chunk = np.zeros(shape=(write_chunk, max_stack_size + max_buffer_size), dtype=np.int32)
        height_chunk = np.zeros(shape=(write_chunk, max_stack_size + max_buffer_size), dtype=np.int32)
        action_count_chunk = np.zeros(shape=(write_chunk, args.num_labels), dtype=np.int32)
        pos_chunk = np.zeros(shape=(write_chunk, max_stack_size + max_buffer_size), dtype=np.int32)
        out_chunk = np.zeros(shape=(write_chunk, max_stack_size + max_buffer_size, args.num_edges), dtype=np.int32)
        inc_chunk = np.zeros(shape=(write_chunk, max_stack_size + max_buffer_size, args.num_edges), dtype=np.int32)
        hist_chunk = np.zeros(shape=(write_chunk, max_hist_size), dtype=np.int32)
        start = True
        chunk_no = 0

        for index, (stack_features, buffer_features) in enumerate(stack_and_buffer_features):
            forms, deps, heads, pos, incoming, outgoing, height = tuple(zip(*(stack_features + buffer_features)))
            index = index % write_chunk
            form_chunk[index] = forms
            dep_chunk[index] = deps
            head_chunk[index] = heads
            pos_chunk[index] = pos
            height_chunk[index] = height

            action_count_chunk[index][previous_action_counts[index]] += 1

            for n, item in enumerate(incoming):
                for id in item:
                    inc_chunk[index, n, id] += 1
            for n, item in enumerate(outgoing):
                for id in item:
                    out_chunk[index, n, id] += 1

            hist_chunk[index] = np.hstack(
                [history_features[index], np.zeros(shape=(max_hist_size - len(history_features[index])))])

            if index % write_chunk == 0 and not start:
                cur_slice = slice(chunk_no * write_chunk, min((chunk_no + 1) * write_chunk, num_examples))

                print("writing to {}".format(cur_slice))

                form_matrix[cur_slice] = form_chunk
                dep_matrix[cur_slice] = dep_chunk
                head_matrix[cur_slice] = head_chunk
                out_matrix[cur_slice] = out_chunk
                inc_matrix[cur_slice] = inc_chunk
                pos_matrix[cur_slice] = pos_chunk
                height_matrix[cur_slice] = height_chunk
                history_matrix[cur_slice] = hist_chunk
                action_counts[cur_slice] = action_count_chunk
                print("Done with {} of {}".format(chunk_no, num_examples // write_chunk))

                chunk_no += 1
            start = False

        if index % write_chunk != 0:
            cur_slice = slice(chunk_no * write_chunk, min((chunk_no + 1) * write_chunk, num_examples))

            print("writing to {}".format(cur_slice))

            form_matrix[cur_slice] = form_chunk[:num_examples % write_chunk]
            dep_matrix[cur_slice] = dep_chunk[:num_examples % write_chunk]
            head_matrix[cur_slice] = head_chunk[:num_examples % write_chunk]
            out_matrix[cur_slice] = out_chunk[:num_examples % write_chunk]
            inc_matrix[cur_slice] = inc_chunk[:num_examples % write_chunk]
            pos_matrix[cur_slice] = pos_chunk[:num_examples % write_chunk]
            height_matrix[cur_slice] = height_chunk[:num_examples % write_chunk]
            history_matrix[cur_slice] = hist_chunk[:num_examples % write_chunk]
            action_counts[cur_slice] = action_count_chunk[:num_examples % write_chunk]
            print("Done with {} of {}".format(chunk_no, num_examples // write_chunk))

        f.create_dataset('history_lengths', data=np.array(history_lengths))
        f.create_dataset('sentence_lengths', data=np.array(sentence_lengths))
        f.create_dataset('state2sent_index', data=np.array(state2passage_id))

        elmo = f.create_group('elmo')
        contextualized_embeddings = embedder.sents2elmo(passage_id2sent)
        for n, emb in enumerate(contextualized_embeddings):
            elmo.create_dataset('{}'.format(n), data=emb, compression="gzip")
        torch.cuda.empty_cache()

    return shapes


def preprocess(args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    print("Processing passages...", file=sys.stderr)

    elmo_embedder = Embedder(args.elmo_path, batch_size=30)

    label_numberer = Numberer()
    pos_numberer = Numberer(first_elements=["<PAD>"])
    dep_numberer = Numberer(first_elements=["<PAD>"])
    edge_numberer = Numberer(first_elements=["<PAD>"])

    print("starting processing training data..", )
    training_data = preprocess_dataset(
        args.training_path,
        args,
        label_numberer=label_numberer,
        pos_numberer=pos_numberer,
        dep_numberer=dep_numberer,
        edge_numberer=edge_numberer,
        train=True
    )
    print("finished processing training data..", )

    args.num_labels = label_numberer.max
    args.num_edges = edge_numberer.max
    args.num_pos = pos_numberer.max
    args.num_deps = dep_numberer.max

    print("...starting to write training features", )
    training_shapes = specific_elmo(training_data, elmo_embedder, args, train=True)
    print("finished writing training data..", )

    # Preprocess validation set
    print("starting to process validation data..", )
    try:
        validation_data = preprocess_dataset(
        args.validation_path,
        args,
        shapes=training_shapes,
        label_numberer=label_numberer,
        pos_numberer=pos_numberer,
        dep_numberer=dep_numberer,
        edge_numberer=edge_numberer,
        train=False
        )
        print("finished processing validation data..", )

        print("..starting to write validation data..", )
        validation_shapes = specific_elmo(validation_data, elmo_embedder, args, train=False)
        print("..finished writing validation data", )

        args.label_list = label_numberer.num2value
        args.dep_list = dep_numberer.num2value
        args.edge_list = dep_numberer.num2value
        args.pos_list = dep_numberer.num2value
        args.shapes = training_shapes

        save_args(args, args.save_dir)
        # Save arguments and dictionaries
        with open(os.path.join(args.save_dir, LABELS_FILENAME), "w", encoding="utf-8") as file:
            label_numberer.to_file(file)
        with open(os.path.join(args.save_dir, DEP_FILENAME), "w", encoding="utf-8") as file:
            dep_numberer.to_file(file)
        with open(os.path.join(args.save_dir, EDGE_FILENAME), "w", encoding="utf-8") as file:
            edge_numberer.to_file(file)
        with open(os.path.join(args.save_dir, POS_FILENAME), "w", encoding="utf-8") as file:
            pos_numberer.to_file(file)
    except:
        import IPython; IPython.embed()

from ucca import constructions

if __name__ == '__main__':
    import sys

    oracle_parser = get_oracle_parser()
    constructions.add_argument(oracle_parser)
    argument_parser = get_preprocess_parser(parents=[oracle_parser])

    args = argument_parser.parse_args(sys.argv[1:])
    preprocess(args)
