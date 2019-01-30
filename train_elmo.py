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
        def float32_variable_storage_getter(getter, name, shape=None, dtype=None,
                                            initializer=None, regularizer=None,
                                            trainable=True,
                                            *args, **kwargs):
            """Custom variable getter that forces trainable variables to be stored in
            float32 precision and then casts them to the training precision.
            """
            storage_dtype = tf.float32 if trainable else dtype
            variable = getter(name, shape, dtype=storage_dtype,
                              initializer=initializer, regularizer=regularizer,
                              trainable=trainable,
                              *args, **kwargs)
            if trainable and dtype != tf.float32:
                variable = tf.cast(variable, dtype)
            return variable
        with tf.variable_scope('model', reuse=False,custom_getter=float32_variable_storage_getter,dtype=tf.float16):
            m = ElModel(args, args.num_labels, num_dependencies=args.num_deps, num_pos=args.num_pos,
                        num_ner=args.num_ner, train=True, predict=False)
        with tf.variable_scope('model', reuse=True,custom_getter=float32_variable_storage_getter,dtype=tf.float16):
            v = ElModel(args, args.num_labels, num_dependencies=args.num_deps, num_pos=args.num_pos,
                        num_ner=args.num_ner, train=False, predict=False)

        gs = tf.train.get_or_create_global_step()
        sv = tf.train.Saver()
        cp = tf.train.latest_checkpoint(os.path.join(args.save_dir, "save_dir"))

        if cp:
            tf.logging.info("Restoring model from latest checkpoint: {}".format(cp))
            import collections
            import re
            def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
                """Compute the union of the current variables and checkpoint variables.
                    taken from bert git repository
                """
                initialized_variable_names = {}
                name_to_variable = collections.OrderedDict()
                for var in tvars:
                    name = var.name
                    m = re.match("^(.*):\\d+$", name)
                    if m is not None:
                        name = m.group(1)
                    name_to_variable[name] = var

                init_vars = tf.train.list_variables(init_checkpoint)

                assignment_map = collections.OrderedDict()
                for x in init_vars:
                    (name, var) = (x[0], x[1])
                    if name not in name_to_variable:
                        continue
                    assignment_map[name] = name
                    initialized_variable_names[name] = 1
                    initialized_variable_names[name + ":0"] = 1

                return (assignment_map, initialized_variable_names)
            tvars = tf.trainable_variables()
            ass_map, initialized_variable_names = get_assignment_map_from_checkpoint(tvars, cp)
            tf.logging.info("**** Trainable Variables ****")
            for var in tvars:
                init_string = ""
                if var.name in initialized_variable_names:
                    init_string = ", *INIT_FROM_CKPT*"
                tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                                init_string)
            tf.train.init_from_checkpoint(cp,ass_map)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

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
            for tn in range(1, args.epoch_steps + 1):
                features, labels = train_q.get()
                feed_dict = dict(zip(train_inputs, features))
                feed_dict[m.labels] = labels
                logits, loss, _, _, tm1, gs = sess.run(
                    [m.logits, m.loss, m.per_class, m.train_op, m.merge,
                     tf.train.get_or_create_global_step()], feed_dict, run_metadata=run_metadata, options=options)
                train_ep_loss += loss.mean()
                acc = np.equal(np.argmax(logits, -1), labels).mean()
                train_ep_acc += acc
                progress.print_network_progress("Training", tn, args.epoch_steps, loss.mean(), train_ep_loss / tn, acc,
                                                train_ep_acc / tn)
                if tn != 0 and tn % 10 == 0:
                    fw.add_summary(tm1, gs)
                    value = tf.Summary.Value(tag="train_acc",simple_value=train_ep_acc / tn)
                    loss = tf.Summary.Value(tag="train_loss", simple_value=train_ep_loss / tn)
                    summary = tf.Summary(value=[value,loss])
                    # fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                    # chrome_trace = fetched_timeline.generate_chrome_trace_format()
                    # with open(os.path.join(args.save_dir,'log_dir/timeline_{}.json'.format(gs)), 'w') as f:
                    #     f.write(chrome_trace)
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
            fw.add_summary(summary, gs)
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
