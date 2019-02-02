import random
import h5py
import numpy as np
import tuepa.finalfrontier as finalfrontier
import ast

def advindexing_roll(A, r):
    rows, column_indices = np.ogrid[:A.shape[0], :A.shape[1]]
    r[r < 0] += A.shape[1]
    column_indices = column_indices - r[:, np.newaxis]
    return A[rows, column_indices]

def aligned(a, alignment=16):
    if (a.ctypes.data % alignment) == 0:
        return a

    extra = alignment // a.itemsize
    buf = np.empty(a.size + extra, dtype=a.dtype)
    ofs = (-buf.ctypes.data % alignment) // a.itemsize
    aa = buf[ofs:ofs+a.size].reshape(a.shape)
    np.copyto(aa, a)
    assert (aa.ctypes.data % alignment) == 0
    return aa

def h5py_worker(data_path, queue, args, batch_size, eval=False):
    """
    Method to read chunks from HDF5 files containing examples.

    Opens a HDF5 file at `data_path`, reads and preprocesses 10 batches at once and sends puts them into `queue`.

    :param data_path: path to HDF5 file.
    :param queue: a queue object
    :param args: named tuple holding commandline arguments and other information
    """
    model = finalfrontier.Model(args.word.finalfrontier, True)
    with h5py.File(data_path, 'r') as data:
        passages = data['passages'][()]
        def prepare(data, index=0):

            state2pid = np.array(data['state2sent_index'])
            if eval:
                index = index * batch_size
                getters = list(range(index, min(index + batch_size, len(state2pid))))
            else:
                index = random.randint(0, len(state2pid) - batch_size)
                getters = list(range(index, min((index + 1) + batch_size, len(state2pid))))

            ids = state2pid[getters]
            batch_sents = passages[ids]

            # for each input chunk cache elmo
            elmos = dict()
            batch_elmo = []

            for n,i in enumerate(ids):
                if i in elmos:
                    batch_elmo.append(elmos[i])
                else:
                    ff_sent = [np.array(model.embedding(t)) for t in ast.literal_eval(batch_sents[n])]
                    res = np.concatenate((data['elmo'][str(i).encode("UTF-8")][()].mean(axis=0),ff_sent),axis=-1)
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
            history_features = advindexing_roll(history_features, history_features.shape[1] - history_lengths)
            ci = data['stack_buffer']['child_indices'][getters]
            m = ci[:,:,0] == 0
            ci[:,:,0][m] = 1
            nonz = ci != 0
            batch_ind = np.argwhere(ci).T[0]
            ci_lengths = np.count_nonzero(ci.reshape((-1, ci.shape[-1])), axis=1)
            ci[:, :, 0][m] = 0
            ci = ci[nonz]
            cei = data['stack_buffer']['child_edge_indices'][getters]
            cei[:,:,0][m] = 1
            nonz = cei != 0
            cei_lengths = np.count_nonzero(cei.reshape((-1, cei.shape[-1])), axis=1)
            cei[:,:,0][m] = 0
            cei = cei[nonz]

            return (form_indices,
                    dep_types,
                    head_indices,
                    pos,
                    ci,
                    ci_lengths,
                    cei,
                    cei_lengths,
                    batch_ind,
                    data['stack_buffer']['ner'][getters],
                    data['stack_buffer']['height'][getters],
                    data['stack_buffer']['in'][getters],
                    data['stack_buffer']['out'][getters],
                    history_features,
                    batch_elmo,
                    data['sentence_lengths'].value[ids],
                    history_lengths,
                    data['action_counts'][getters],
                    data['action_ratios'][getters],
                    data['node_ratios'][getters],
                    data['stack_buffer']['root'][getters]), \
                   data['labels'][getters]


        n_rows = len(data['labels'])
        index = 0
        next_b = prepare(data, index=0)
        index += 1
        while True:
            if next_b:
                queue.put(next_b)
                next_b = prepare(data, index=index)
                index += 1
                if index == n_rows // batch_size:
                    next_b = prepare(data, index=index)
                    index = 0
