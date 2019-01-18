from tuepa.data.preprocessing import read_passages, Shapes
from tuepa.parser import State, Oracle

import h5py_cache
import numpy as np
import torch
from ucca.layer1 import EdgeTags


def squash_singleton_terminals(passage):
    nodes_to_remove = []

    for node in passage._nodes.values():
        # Find nodes with only single Terminal edges to squash
        if len(node.outgoing) == 1 and node.outgoing[0].tag == EdgeTags.Terminal:
            token_node = node.outgoing[0].child

            # Relink incoming edges
            for edge in node.incoming:
                parent = edge.parent
                parent.remove(edge)
                parent.add(edge.tag, token_node, edge_attrib=edge.attrib)

            nodes_to_remove.append(node)

    # Destroy orphan nodes
    for node in nodes_to_remove:
        node.destroy()


def extract_elmo_features(args, state, label_numberer, dep_numberer, pos_numberer, ner_numberer, edge_numberer,
                          train=True):
    stack_features = []
    buffer_features = []
    stack = state.stack
    buffer = state.buffer

    def null_features():
        return [0, 0, 0, 0, 0, [], [], 0, 0, []]

    def extract_feature(node):
        if node.text is not None:
            form = node.index + 1
            dep = node.extra['dep']
            head = node.extra['head'] + form
            pos = node.extra['tag']
            ner = node.extra['ent_type']
            child_indices = []
            root = 0
        else:
            form = 1
            dep = "<NT>"
            head = 1
            pos = "<NT>"
            ner = "<NT>"
            child_indices = [t.index + 1 for t in node.terminals]
            root = int(node.is_root)

        incoming = [edge_numberer.number("{}-{}".format(e.tag, "remote" if e.remote else "primary"), train) for e in node.incoming]
        outgoing = [edge_numberer.number("{}-{}".format(e.tag, "remote" if e.remote else "primary"), train) for e in node.outgoing]

        height = node.height
        return [form, dep_numberer.number(dep, train=train), head, pos_numberer.number(pos, train=train),
                ner_numberer.number(ner, train=train), incoming,
                outgoing, height, root, child_indices]

    for n in range(args.stack_elements):
        try:
            node = stack[-n]
            stack_features.append(extract_feature(node))

            def try_or_value(val, sequence, ind):
                try:
                    return sequence[ind]
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

            if len(node.parents) > 1:
                right_parent = try_or_value(0, node.parents, -1)
                right_parent = try_or_function(test_fn, null_features, extract_feature, right_parent)
            else:
                right_parent = null_features()

            stack_features.append(right_parent)

            left_child = try_or_value(0, node.children, 0)
            left_child = try_or_function(test_fn, null_features, extract_feature, left_child)
            stack_features.append(left_child)

            if len(node.children) > 1:
                right_child = try_or_value(0, node.children, -1)
                right_child = try_or_function(test_fn, null_features, extract_feature, right_child)
            else:
                right_child = null_features()
            stack_features.append(right_child)

        except IndexError:
            stack_features.append(null_features())
            stack_features.append(null_features())
            stack_features.append(null_features())
            stack_features.append(null_features())
            stack_features.append(null_features())
    for n in range(args.buffer_elements):
        try:
            buffer_features.append(extract_feature(buffer[n]))
        except IndexError:
            buffer_features.append(null_features())

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
                       ner_numberer=None,
                       train=False):
    if not train:
        max_stack_size = shapes.max_stack_size
        max_buffer_size = shapes.max_buffer_size

    else:
        max_stack_size = max_buffer_size = -1

    stack_and_buffer_features = []
    previous_action_counts = []
    labels = []
    node_ratios = []
    action_ratios = []
    history_features = []
    sentence_lengths = []
    history_lengths = []
    passage_id2sent = []
    state2passage_id = []
    passage_id = 0
    passage_names = []

    for passage in read_passages([path]):
        if args.squash_singleton_terminals:
            squash_singleton_terminals(passage)

        sentence = [str(n) for n in passage.layer("0").all]
        if len(sentence) > 100:
            continue
        passage_names.append(passage.ID)
        passage_id2sent.append(sentence)
        sentence_lengths.append(len(sentence))

        state = State(passage, args)
        oracle = Oracle(passage, args)
        state_no = 0
        passage_actions = []
        while not state.finished:
            node_ratios.append(state.node_ratio())
            action_ratios.append(state.action_ratio())
            state2passage_id.append(passage_id)
            actions = oracle.generate_actions(state=state)
            action = next(actions)

            stack_features, buffer_features, state_history = extract_elmo_features(
                args,
                state,
                label_numberer,
                pos_numberer=pos_numberer,
                dep_numberer=dep_numberer,
                edge_numberer=edge_numberer,
                ner_numberer=ner_numberer,
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

    return stack_and_buffer_features, Shapes(max_stack_size,
                                             max_buffer_size), history_features, history_lengths, state2passage_id, passage_id2sent, sentence_lengths, previous_action_counts, action_ratios, node_ratios, labels, passage_names


def specific_elmo(features, embedder, args, train, write_chunk=8192):
    stack_and_buffer_features, shapes, history_features, history_lengths, state2passage_id, passage_id2sent, sentence_lengths, previous_action_counts, action_ratios, node_ratios, labels, passage_names = features
    max_stack_size = shapes.max_stack_size
    max_buffer_size = shapes.max_buffer_size
    max_hist_size = np.max(history_lengths)
    num_examples = len(stack_and_buffer_features)

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
        child_matrix = s_b.create_dataset('child_indices', shape=(num_examples, max_stack_size + max_buffer_size, 30),
                                          dtype=np.int32, fillvalue=0,
                                          maxshape=(num_examples, max_stack_size + max_buffer_size, None))
        height_matrix = s_b.create_dataset('height', shape=(num_examples, max_stack_size + max_buffer_size),
                                           dtype=np.int32, fillvalue=0)
        root_matrix = s_b.create_dataset('root', shape=(num_examples, max_stack_size + max_buffer_size),
                                         dtype=np.int32, fillvalue=0)
        ner_matrix = s_b.create_dataset('ner', shape=(num_examples, max_stack_size + max_buffer_size),
                                        dtype=np.int32, fillvalue=0)
        pos_matrix = s_b.create_dataset('pos', shape=(num_examples, max_stack_size + max_buffer_size),
                                        dtype=np.int32, fillvalue=0)
        out_matrix = s_b.create_dataset('out', shape=(
            num_examples, max_stack_size + max_buffer_size, args.num_edges),
                                        dtype=np.int32, fillvalue=0)
        inc_matrix = s_b.create_dataset('in', shape=(
            num_examples, max_stack_size + max_buffer_size, args.num_edges),
                                        dtype=np.int32, fillvalue=0)
        action_counts = f.create_dataset('action_counts', shape=(num_examples, args.num_labels), dtype=np.int32)
        history_matrix = f.create_dataset('history_features', shape=(num_examples, max_hist_size),
                                          dtype=np.int32, fillvalue=0)

        form_chunk = np.zeros(shape=(write_chunk, max_stack_size + max_buffer_size), dtype=np.int32)
        dep_chunk = np.zeros(shape=(write_chunk, max_stack_size + max_buffer_size), dtype=np.int32)
        head_chunk = np.zeros(shape=(write_chunk, max_stack_size + max_buffer_size), dtype=np.int32)
        child_chunk = np.zeros(shape=(write_chunk, max_stack_size + max_buffer_size, child_matrix.shape[-1]),
                               dtype=np.int32)
        height_chunk = np.zeros(shape=(write_chunk, max_stack_size + max_buffer_size), dtype=np.int32)
        root_chunk = np.zeros(shape=(write_chunk, max_stack_size + max_buffer_size), dtype=np.int32)
        ner_chunk = np.zeros(shape=(write_chunk, max_stack_size + max_buffer_size), dtype=np.int32)
        action_count_chunk = np.zeros(shape=(write_chunk, args.num_labels), dtype=np.int32)
        pos_chunk = np.zeros(shape=(write_chunk, max_stack_size + max_buffer_size), dtype=np.int32)
        out_chunk = np.zeros(shape=(write_chunk, max_stack_size + max_buffer_size, args.num_edges), dtype=np.int32)
        inc_chunk = np.zeros(shape=(write_chunk, max_stack_size + max_buffer_size, args.num_edges), dtype=np.int32)
        hist_chunk = np.zeros(shape=(write_chunk, max_hist_size), dtype=np.int32)
        start = True
        chunk_no = 0
        max_n = -1
        for ex_index, (stack_features, buffer_features) in enumerate(stack_and_buffer_features):
            if ex_index % write_chunk == 0 and not start:
                cur_slice = slice(chunk_no * write_chunk, min((chunk_no + 1) * write_chunk, num_examples))

                print("writing to {}".format(cur_slice))
                form_matrix[cur_slice] = form_chunk
                dep_matrix[cur_slice] = dep_chunk
                head_matrix[cur_slice] = head_chunk
                child_matrix[cur_slice] = child_chunk
                out_matrix[cur_slice] = out_chunk
                inc_matrix[cur_slice] = inc_chunk
                ner_matrix[cur_slice] = ner_chunk
                pos_matrix[cur_slice] = pos_chunk
                root_matrix[cur_slice] = root_chunk
                height_matrix[cur_slice] = height_chunk
                history_matrix[cur_slice] = hist_chunk
                action_counts[cur_slice] = action_count_chunk

                max_n = max((out_chunk.max(), inc_chunk.max(), action_count_chunk.max(), max_n))

                form_chunk = np.zeros(shape=(write_chunk, max_stack_size + max_buffer_size), dtype=np.int32)
                dep_chunk = np.zeros(shape=(write_chunk, max_stack_size + max_buffer_size), dtype=np.int32)
                head_chunk = np.zeros(shape=(write_chunk, max_stack_size + max_buffer_size), dtype=np.int32)
                child_chunk = np.zeros(shape=(write_chunk, max_stack_size + max_buffer_size, child_matrix.shape[-1]),
                                       dtype=np.int32)
                height_chunk = np.zeros(shape=(write_chunk, max_stack_size + max_buffer_size), dtype=np.int32)
                root_chunk = np.zeros(shape=(write_chunk, max_stack_size + max_buffer_size), dtype=np.int32)
                ner_chunk = np.zeros(shape=(write_chunk, max_stack_size + max_buffer_size), dtype=np.int32)
                action_count_chunk = np.zeros(shape=(write_chunk, args.num_labels), dtype=np.int32)
                pos_chunk = np.zeros(shape=(write_chunk, max_stack_size + max_buffer_size), dtype=np.int32)
                out_chunk = np.zeros(shape=(write_chunk, max_stack_size + max_buffer_size, args.num_edges),
                                     dtype=np.int32)
                inc_chunk = np.zeros(shape=(write_chunk, max_stack_size + max_buffer_size, args.num_edges),
                                     dtype=np.int32)
                hist_chunk = np.zeros(shape=(write_chunk, max_hist_size), dtype=np.int32)
                print("Done with {} of {}".format(chunk_no, num_examples // write_chunk))

                chunk_no += 1

            start = False
            forms, deps, heads, pos, ner, incoming, outgoing, height, root, children = tuple(
                zip(*(stack_features + buffer_features)))

            index = ex_index % write_chunk
            form_chunk[index] = forms
            dep_chunk[index] = deps
            head_chunk[index] = heads
            pos_chunk[index] = pos
            ner_chunk[index] = ner
            height_chunk[index] = height
            root_chunk[index] = root

            for n,child in enumerate(children):
                for k, c in enumerate(child[:30]):
                    child_chunk[index,n,k] = c

            for action in previous_action_counts[ex_index]:
                action_count_chunk[index, action] += 1

            for n, item in enumerate(incoming):
                for id in item:
                    inc_chunk[index, n, id] += 1

            for n, item in enumerate(outgoing):
                for id in item:
                    out_chunk[index, n, id] += 1

            hist_chunk[index] = np.hstack(
                [history_features[index], np.zeros(shape=(max_hist_size - len(history_features[index])))])

        if ex_index % write_chunk != 0:
            cur_slice = slice(chunk_no * write_chunk, min((chunk_no + 1) * write_chunk, num_examples))

            print("writing to {}".format(cur_slice))

            form_matrix[cur_slice] = form_chunk[:num_examples % write_chunk]
            dep_matrix[cur_slice] = dep_chunk[:num_examples % write_chunk]
            head_matrix[cur_slice] = head_chunk[:num_examples % write_chunk]
            out_matrix[cur_slice] = out_chunk[:num_examples % write_chunk]
            inc_matrix[cur_slice] = inc_chunk[:num_examples % write_chunk]
            ner_matrix[cur_slice] = ner_chunk[:num_examples % write_chunk]
            pos_matrix[cur_slice] = pos_chunk[:num_examples % write_chunk]
            root_matrix[cur_slice] = root_chunk[:num_examples % write_chunk]
            height_matrix[cur_slice] = height_chunk[:num_examples % write_chunk]
            history_matrix[cur_slice] = hist_chunk[:num_examples % write_chunk]
            action_counts[cur_slice] = action_count_chunk[:num_examples % write_chunk]
            max_n = max((out_chunk.max(), inc_chunk.max(), action_count_chunk.max(), max_n))
            print("Done with {} of {}".format(chunk_no, num_examples // write_chunk))

        f.create_dataset('history_lengths', data=np.array(history_lengths))
        f.create_dataset('sentence_lengths', data=np.array(sentence_lengths))
        f.create_dataset('state2sent_index', data=np.array(state2passage_id))

        f.create_dataset('action_ratios', data=np.array(action_ratios))
        f.create_dataset('node_ratios', data=np.array(node_ratios))
        import h5py
        dt = h5py.special_dtype(vlen=str)
        f.create_dataset('passage_names', shape=(len(passage_id2sent),), dtype=dt)
        f['passage_names'][()] = passage_names
        f.create_dataset('labels', data=np.array(labels))
        f.create_dataset('passages', shape=(len(passage_id2sent),),dtype=dt)
        f['passages'][()] = passage_id2sent

        elmo = f.create_group('elmo')
        contextualized_embeddings = embedder.sents2elmo(passage_id2sent)

        for n, emb in enumerate(contextualized_embeddings):
            elmo.create_dataset('{}'.format(n), data=emb, compression="gzip")
        torch.cuda.empty_cache()

    return shapes, max_n
