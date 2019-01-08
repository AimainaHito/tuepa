import multiprocessing

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
             data['stack_buffer']['form_indices'][0].shape,  # heights
             data['stack_buffer']['form_indices'][0].shape + (args.num_edges,),  # inc
             data['stack_buffer']['form_indices'][0].shape + (args.num_edges,),  # out
             tf.TensorShape([None]),  # history
             tf.TensorShape([None, 1024]),  # elmo
             tf.TensorShape([]),  # sentence lengths
             tf.TensorShape([]),  # history lengths
             tf.TensorShape([args.num_labels])), # action counts
            tf.TensorShape([]))  # labels
    q = multiprocessing.Queue(maxsize=4)
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
        ((tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.float32, tf.int32,
          tf.int32, tf.int32), tf.int32))
        if train:
            return d.shuffle(args.batch_size * 20)
        else:
            return d.shuffle(args.batch_size * 20)

    if train_or_eval:
        return lambda: get_dataset().padded_batch(args.batch_size, data_shapes, drop_remainder=True).prefetch(1)
    else:
        pass


def h5py_worker(data_path, queue, args):
    """
    Method to read chunks from HDF5 files containing examples.

    Opens a HDF5 file at `data_path`, reads and preprocesses 10 batches at once and sends puts them into `queue`.

    :param data_path: path to HDF5 file.
    :param queue: a queue object
    :param args: named tuple holding commandline arguments and other information
    """

    def prepare(data, index=None):
        state2pid = np.array(data['state2sent_index'])
        getters = list(range(index * args.batch_size * 20, min((index + 1) * args.batch_size * 20, len(state2pid))))

        ids = state2pid[getters]

        # for each input chunk cache elmo
        elmos = dict()
        batch_elmo = []
        for i in ids:
            if i in elmos:
                batch_elmo.append(elmos[i])
            else:
                res = data['elmo'][str(i).encode("UTF-8")].value
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
        return form_indices, \
               dep_types, \
               head_indices, \
               pos, \
               data['stack_buffer']['height'][getters], \
               data['stack_buffer']['in'][getters], \
               data['stack_buffer']['out'][getters], \
               data['history_features'][getters], \
               batch_elmo, \
               data['sentence_lengths'].value[ids], \
               data['history_lengths'][getters], \
               data['action_counts'][getters],\
               data['labels'][getters]

    with h5py.File(data_path, 'r') as data:
        max_ind = len(data['labels'])
        index = 0
        next_b = prepare(data, index=index)
        while True:
            if next_b:
                queue.put(next_b)
                try:
                    next_b = prepare(data, index=index)
                except:
                    index = 0
                    next_b = prepare(data, index=index)
                index += 1
                if index == max_ind - args.batch_size * 20:
                    index = 0