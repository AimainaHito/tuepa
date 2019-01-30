import multiprocessing
import os

import tensorflow as tf
from argparse import Namespace
from timeit import default_timer

from tuepa.util.config import save_args, load_args, LABELS_FILENAME, \
    DEP_FILENAME, EDGE_FILENAME, POS_FILENAME, NER_FILENAME
from tuepa.nn import ElModel
from tuepa.util.numberer import load_numberer_from_file
from tuepa.util import SaverHook, PerClassHook
from tuepa.data.elmo.elmo_input import h5py_worker
import numpy as np
import tuepa.progress as progress


def train(args):

    train_q = multiprocessing.Queue(maxsize=100)
    h5py_worker(args.training_path, train_q, args, args.batch_size)
    val_q = multiprocessing.Queue(maxsize=50)
    train_p = multiprocessing.Process(target=h5py_worker, args=(args.training_path, train_q, args,args.batch_size))
    val_p = multiprocessing.Process(target=h5py_worker, args=(args.validation_path, val_q, args, 512,True))
    import h5py
    with h5py.File(args.validation_path, "r") as f:
        val_rows = len(f['labels'])
    train_p.daemon = True
    val_p.daemon = True
    train_p.start()
    val_p.start()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    from tensorflow.python.client import timeline

    with tf.Session(config=config) as sess:
        with tf.variable_scope('model', reuse=False):
            m = ElModel(args, args.num_labels, num_dependencies=args.num_deps, num_pos=args.num_pos,
                        num_ner=args.num_ner, train=True, predict=False)
        with tf.variable_scope("model", reuse=True):
            v = ElModel(args, args.num_labels, num_dependencies=args.num_deps, num_pos=args.num_pos,
                        num_ner=args.num_ner, train=False, predict=False)

        gs = tf.train.get_or_create_global_step()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        sv = tf.train.Saver()
        cp = tf.train.latest_checkpoint(os.path.join(args.save_dir, "save_dir"))
        if cp:
            tf.logging.info("Restoring model from latest checkpoint: {}".format(cp))
            sv.restore(sess, cp)

        train_inputs = m.inpts
        val_inputs = v.inpts
        fw = tf.summary.FileWriter(logdir=os.path.join(args.save_dir, "log_dir"), graph=sess.graph)
        steps = val_rows // 512
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata= tf.RunMetadata()

        for iteration in range(100000):
            start_time = default_timer()
            train_ep_loss = 0
            train_ep_acc = 0
            for tn in range(1, steps + 1):
                features, labels = train_q.get()
                feed_dict = dict(zip(train_inputs, features))
                feed_dict[m.labels] = labels
                logits, loss, _, _, tm1, gs = sess.run(
                    [m.logits, m.loss, m.per_class, m.train_op, m.merge,
                     tf.train.get_or_create_global_step()], feed_dict, run_metadata=run_metadata, options=options)
                train_ep_loss += loss.mean()
                acc = np.equal(np.argmax(logits, -1), labels).mean()
                train_ep_acc += acc
                progress.print_network_progress("Training", tn, steps, loss.mean(), train_ep_loss / tn, acc,
                                                train_ep_acc / tn)
                if tn != 0 and tn % 10 == 0:
                    fw.add_summary(tm1, gs)
                    value = tf.Summary.Value(tag="train_acc",simple_value=train_ep_acc / tn)
                    loss = tf.Summary.Value(tag="train_loss", simple_value=train_ep_loss / tn)
                    summary = tf.Summary(value=[value,loss])
                    fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                    chrome_trace = fetched_timeline.generate_chrome_trace_format()
                    with open(os.path.join(args.save_dir,'log_dir/timeline_{}.json'.format(gs)), 'w') as f:
                        f.write(chrome_trace)
                    fw.add_run_metadata(run_metadata,tag="train_meta_{}".format(gs),global_step=gs)
                    fw.add_summary(summary,gs)
                    fw.flush()


            val_ep_loss = 0
            val_ep_accuracy = 0
            val_ep_mean_per_class = 0

            for n in range(1, steps + 1):
                features, labels = val_q.get()
                feed_dict = dict(zip(val_inputs, features))
                feed_dict[v.labels] = labels
                logits, loss, maccurcy, mpc = sess.run(
                    [v.logits, v.loss, v.per_class,v.mpc], feed_dict)
                acc = np.equal(np.argmax(logits, -1), labels).mean()
                val_ep_loss += loss.mean()
                val_ep_accuracy += acc
                val_ep_mean_per_class += mpc

                progress.print_network_progress("Validation", n, steps, loss.mean(), val_ep_loss / n, acc,
                                                val_ep_accuracy / n)

            value = tf.Summary.Value(tag="val_acc",simple_value=val_ep_accuracy/n)
            per_class = tf.Summary.Value(tag="mean_per_class",simple_value=val_ep_mean_per_class/n)
            loss = tf.Summary.Value(tag="val_loss", simple_value=val_ep_loss / n)
            summary = tf.Summary(value=[value, per_class, loss])
            fw.add_summary(summary, gs + n)
            fw.flush()


            save_name = "tuepa_{}.ckpt".format(gs)
            s = sv.save(sess, os.path.join(args.save_dir, "save_dir", save_name))
            tf.logging.info("Saved {}".format(s))
            progress.print_iteration_info(
                iteration,
                train_ep_loss,
                train_ep_acc / tn,
                val_ep_loss,
                val_ep_accuracy / n,
                start_time,
                args.log_file
            )


def main(args):
    import tuepa.util.config as config
    tf.logging.set_verbosity(tf.logging.INFO)
    argument_parser = config.get_elmo_parser()
    args = argument_parser.parse_args(args)
    model_args = load_args(args.save_dir)
    # Merge preprocess args with train args
    args = Namespace(**{**vars(model_args), **vars(args)})

    with open(os.path.join(args.save_dir, LABELS_FILENAME), "r", encoding="utf-8") as file:
        label_numberer = load_numberer_from_file(file)

    with open(os.path.join(args.save_dir, EDGE_FILENAME), "r", encoding="utf-8") as file:
        edge_numberer = load_numberer_from_file(file)

    with open(os.path.join(args.save_dir, DEP_FILENAME), "r", encoding="utf-8") as file:
        dep_numberer = load_numberer_from_file(file)

    with open(os.path.join(args.save_dir, POS_FILENAME), "r", encoding="utf-8") as file:
        pos_numberer = load_numberer_from_file(file)

    with open(os.path.join(args.save_dir, NER_FILENAME), "r", encoding="utf-8") as file:
        ner_numberer = load_numberer_from_file(file)

    # save args for eval call
    save_args(args, args.save_dir)

    train(args=args)


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
