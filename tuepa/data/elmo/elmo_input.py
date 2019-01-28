import multiprocessing
import random

import h5py

import numpy as np
import tensorflow as tf

def get_elmo_input_fn(data_path, train_or_eval, args, train):
    """
    Returns the input_fn for the elmo model. Starts a background process that reads and preprocesses chunks from
    HDF5 files.

    Feature ordering: (stack_buffer, history, elmo, padding, nonterminal,sentence_lens, history_lens)

    :param data_shapes: tuple describing containing shape information of input tensors
    :param batch_size: mini batch size.
    :param train_or_eval: train and eval input fn are identical while predict differs.
    :param train: indicator if training is running. Training means shuffled input.
    :return: a callable which returns a tf.Dataset build from a generator.
    """
    with h5py.File(args.training_path, 'r') as data:
        data_shapes = (
            (data['stack_buffer']['form_indices'][0].shape,  # stack and buffer
             data['stack_buffer']['form_indices'][0].shape,
             data['stack_buffer']['form_indices'][0].shape,
             data['stack_buffer']['form_indices'][0].shape,
             data['stack_buffer']['form_indices'][0].shape + (60,), # child nodes
             data['stack_buffer']['form_indices'][0].shape + (60,), # child types 
             data['stack_buffer']['form_indices'][0].shape, # ner
             data['stack_buffer']['form_indices'][0].shape,  # heights
             data['stack_buffer']['form_indices'][0].shape + (args.num_edges,),  # inc
             data['stack_buffer']['form_indices'][0].shape + (args.num_edges,),  # out
             tf.TensorShape([None]),  # history
             tf.TensorShape([None, 1024]),  # elmo
             tf.TensorShape([]),  # sentence lengths
             tf.TensorShape([]),  # history lengths
             tf.TensorShape([args.num_labels]),
             tf.TensorShape([]),  # action_ratio
             tf.TensorShape([]),# node ratio
             data['stack_buffer']['form_indices'][0].shape,), # root
            tf.TensorShape([]))  # labels
    q = multiprocessing.Queue(maxsize=25)
    p = multiprocessing.Process(target=h5py_worker, args=(data_path, q, args))
    p.daemon = True
    p.start()

    def get_dataset():
        def generator():
            # listen forever
            while True:
                for item in zip(*q.get()):
                    # ((features), labels)
                    yield tuple((item[:-1], item[-1]))

        d = tf.data.Dataset.from_generator(generator, output_types=
        # ((form_indices, dep_types, head_indices, pos, height,inc, out, history, elmo, sent_lens, hist_lens),labels)
        ((tf.int32, tf.int32, tf.int32,tf.int32, tf.int32,tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.float32, tf.int32,
          tf.int32, tf.int32, tf.float32, tf.float32,tf.int32), tf.int32))
        if train:
            return d.shuffle(args.batch_size * 5)
        else:
            return d.shuffle(args.batch_size * 5)

    if train_or_eval:
        return lambda: get_dataset().padded_batch(args.batch_size, data_shapes, drop_remainder=True).prefetch(1)
    else:
        pass

def advindexing_roll(A, r):
    rows, column_indices = np.ogrid[:A.shape[0], :A.shape[1]]    
    r[r < 0] += A.shape[1]
    column_indices = column_indices - r[:,np.newaxis]
    return A[rows, column_indices]

def h5py_worker(data_path, queue, args):
    """
    Method to read chunks from HDF5 files containing examples.

    Opens a HDF5 file at `data_path`, reads and preprocesses 10 batches at once and sends puts them into `queue`.

    :param data_path: path to HDF5 file.
    :param queue: a queue object
    :param args: named tuple holding commandline arguments and other information
    """

    def prepare(data):
        state2pid = np.array(data['state2sent_index'])

        index = random.randint(0, len(state2pid)-(args.batch_size*1))

        getters = list(range(index, min((index + 1) + args.batch_size * 1, len(state2pid))))
        ids = state2pid[getters]

        # for each input chunk cache elmo
        elmos = dict()
        batch_elmo = []

        for i in ids:
            if i in elmos:
                batch_elmo.append(elmos[i])
            else:
                res = data['elmo'][str(i).encode("UTF-8")].value[2]#, [1, 0, 2])
                elmos[i] = res
                batch_elmo.append(elmos[i])

        batch_elmo = np.array(batch_elmo)
        max_length = max(map(len, batch_elmo))
        # loop over sentences and pad them to max length in batch
        batch_elmo = [np.vstack([n, np.zeros(
            shape=[max_length - len(n),
                   n.shape[-1]],
            dtype=np.float32)])
                      for n in batch_elmo]

        form_indices = data['stack_buffer']['form_indices'][getters]
        dep_types = data['stack_buffer']['dependencies'][getters]
        head_indices = data['stack_buffer']['head_indices'][getters]
        pos = data['stack_buffer']['pos'][getters]
        history_lengths = data['history_lengths'][getters]
        history_features = data['history_features'][getters]
        history_features = advindexing_roll(history_features,history_features.shape[1]-history_lengths)

        return form_indices, \
               dep_types, \
               head_indices, \
               pos, \
               data['stack_buffer']['child_indices'][getters][:,:,:60], \
               data['stack_buffer']['child_edge_indices'][getters][:,:,:60], \
               data['stack_buffer']['ner'][getters],\
               data['stack_buffer']['height'][getters], \
               data['stack_buffer']['in'][getters], \
               data['stack_buffer']['out'][getters], \
               history_features, \
               batch_elmo, \
               data['sentence_lengths'].value[ids], \
               history_lengths, \
               data['action_counts'][getters], \
               data['action_ratios'][getters], \
               data['node_ratios'][getters], \
               data['stack_buffer']['root'][getters],\
               data['labels'][getters]

    with h5py.File(data_path, 'r') as data:
        next_b = prepare(data)
        while True:
            if next_b:
                queue.put(next_b)
                next_b = prepare(data)
