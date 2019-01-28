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
    train_q = multiprocessing.Queue(maxsize=5)
    val_q = multiprocessing.Queue(maxsize=5)
    train_p = multiprocessing.Process(target=h5py_worker, args=(args.training_path, train_q, args))
    val_p = multiprocessing.Process(target=h5py_worker, args=(args.validation_path, val_q, args))
    train_p.daemon = True
    val_p.daemon = True
    train_p.start()
    val_p.start()

    with tf.Session() as sess:
        with tf.variable_scope('model', reuse=False):
            m = ElModel(args, args.num_labels, num_dependencies=args.num_deps, num_pos=args.num_pos,
                        num_ner=args.num_ner, train=True, predict=False)
        with tf.variable_scope("model", reuse=True):
            v = ElModel(args, args.num_labels, num_dependencies=args.num_deps, num_pos=args.num_pos,
                        num_ner=args.num_ner, train=False, predict=False)

            hooks = [SaverHook(
                labels=args.label_list,
                confusion_matrix_tensor_name='model_1/mean_iou/total_confusion_matrix',
                summary_writer=tf.summary.FileWriterCache.get(os.path.join(args.save_dir, "eval_validation"))),
                PerClassHook(
                    labels=args.label_list,
                    tensor_name='model_1/mean_accuracy/div_no_nan',
                    summary_writer=tf.summary.FileWriterCache.get(os.path.join(args.save_dir, "eval_validation"))),
            ]
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        sv = tf.train.Saver()
        cp = tf.train.latest_checkpoint(os.path.join(args.save_dir,"save_dir"))
        if cp:
            tf.logging.info("Restoring model from latest checkpoint: {}".format(cp))
            sv.restore(sess, cp)

        train_inputs = m.inpts
        val_inputs = v.inpts
        fw = tf.summary.FileWriter(logdir=os.path.join(args.save_dir, "log_dir"), graph=sess.graph)
        for iteration in range(10000):
            start_time = default_timer()
            train_ep_loss = 0
            train_ep_acc = 0
            for n in range(1,args.epoch_steps+1):
                features, labels = train_q.get()
                feed_dict = dict(zip(train_inputs, features))
                feed_dict[m.labels] = labels
                logits, loss, _,_,_,_,tm1, gs = sess.run(
                    [m.logits, m.loss,m.accuracy,m.per_class,m.mean_iou, m.train_op,m.merge, tf.train.get_or_create_global_step()], feed_dict)
                fw.add_summary(tm1,gs)
                train_ep_loss += loss.mean()
                acc = np.equal(np.argmax(logits,-1),labels).mean()
                train_ep_acc += acc
                progress.print_network_progress("Training", n,args.epoch_steps,loss.mean(),train_ep_loss/n,acc,train_ep_acc/n )


            print(np.mean(train_ep_loss))

            val_ep_loss = 0
            val_ep_accuracy = 0
            for n in range(1,args.epoch_steps+1):
                features, labels = val_q.get()
                feed_dict = dict(zip(val_inputs, features))
                feed_dict[v.labels] = labels
                logits, loss, mer, accuracy, maccurcy, mious, gs = sess.run(
                    [v.logits, v.loss, v.merge, v.accuracy, v.per_class, v.mean_iou, tf.train.get_or_create_global_step()], feed_dict)
                acc = np.equal(np.argmax(logits, -1), labels).mean()
                val_ep_loss += loss.mean()
                val_ep_accuracy += acc
                progress.print_network_progress("Validation", n, args.epoch_steps, loss.mean(), val_ep_loss/n, acc, val_ep_accuracy/n)
            fw.add_summary(mer, gs)
            for hook in hooks:
                hook.end(sess)
            save_name = "tuepa_{}.ckpt".format(gs)
            s = sv.save(sess, os.path.join(args.save_dir,"save_dir", save_name))
            tf.logging.info("Saved {}".format(s))
            progress.print_iteration_info(
                iteration,
                train_ep_loss,
                train_ep_acc/n,
                val_ep_loss,
                val_ep_accuracy/n,
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
