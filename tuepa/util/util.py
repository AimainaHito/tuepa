import textwrap
import re
import io
import itertools

import tensorflow as tf
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
import random

class SaverHook(tf.train.SessionRunHook):
    """
    Taken from: https://github.com/tensorflow/tensorboard/issues/227
    Saves a confusion matrix as a Summary so that it can be shown in tensorboard
    """

    def __init__(self, labels, confusion_matrix_tensor_name, summary_writer):
        """Initializes a `SaveConfusionMatrixHook`.

        :param labels: Iterable of String containing the labels to print for each
                       row/column in the confusion matrix.
        :param confusion_matrix_tensor_name: The name of the tensor containing the confusion
                                             matrix
        :param summary_writer: The summary writer that will save the summary
        """
        self.confusion_matrix_tensor_name = confusion_matrix_tensor_name
        self.labels = labels
        self._summary_writer = summary_writer

    def end(self, session):
        cm = tf.get_default_graph().get_tensor_by_name(
            self.confusion_matrix_tensor_name + ':0').eval(session=session).astype(int)
        globalStep = tf.train.get_global_step().eval(session=session)
        figure = self._plot_confusion_matrix(cm)
        summary = self._figure_to_summary(figure)
        self._summary_writer.add_summary(summary, globalStep)

    def _figure_to_summary(self, fig):
        """
        Converts a matplotlib figure ``fig`` into a TensorFlow Summary object
        that can be directly fed into ``Summary.FileWriter``.
        :param fig: A ``matplotlib.figure.Figure`` object.
        :return: A TensorFlow ``Summary`` protobuf object containing the plot image
                 as a image summary.
        """

        # attach a new canvas if not exists
        if fig.canvas is None:
            matplotlib.backends.backend_agg.FigureCanvasAgg(fig)

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()

        # get PNG data from the figure
        png_buffer = io.BytesIO()
        fig.canvas.print_png(png_buffer)
        png_encoded = png_buffer.getvalue()
        png_buffer.close()

        summary_image = tf.Summary.Image(height=h, width=w, colorspace=4,  # RGB-A
                                         encoded_image_string=png_encoded)
        summary = tf.Summary(value=[tf.Summary.Value(tag=self.confusion_matrix_tensor_name, image=summary_image)])
        return summary

    def _plot_confusion_matrix(self, cm):
        '''
        :param cm: A confusion matrix: A square ```numpy array``` of the same size as self.labels
    `   :return:  A ``matplotlib.figure.Figure`` object with a numerical and graphical representation of the cm array
        '''
        numClasses = len(self.labels)

        fig = plt.Figure(figsize=(numClasses, numClasses), dpi=100, facecolor='w', edgecolor='k')
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(cm, cmap='Oranges')

        classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x) for x in self.labels]
        classes = ['\n'.join(textwrap.wrap(l, 20)) for l in classes]

        tick_marks = np.arange(len(classes))

        ax.set_xlabel('Predicted')
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(classes, rotation=-90, ha='center')
        ax.xaxis.set_label_position('bottom')
        ax.xaxis.tick_bottom()

        ax.set_ylabel('True Label')
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(classes, va='center')
        ax.yaxis.set_label_position('left')
        ax.yaxis.tick_left()

        for i, j in itertools.product(range(numClasses), range(numClasses)):
            ax.text(j, i, int(cm[i, j]) if cm[i, j] != 0 else '.', horizontalalignment="center",
                    verticalalignment='center', color="black")
        fig.set_tight_layout(True)
        return fig

from tuepa.parser import parser

class EvalHook(tf.train.SessionRunHook):
    """
    Taken from: https://github.com/tensorflow/tensorboard/issues/227
    Saves a confusion matrix as a Summary so that it can be shown in tensorboard
    """

    def __init__(self, args, logits, inputs, passages,summary_writer):
        """Initializes a `SaveConfusionMatrixHook`.

        :param labels: Iterable of String containing the labels to print for each
                       row/column in the confusion matrix.
        :param confusion_matrix_tensor_name: The name of the tensor containing the confusion
                                             matrix
        :param summary_writer: The summary writer that will save the summary
        """
        args.model_type = "elmo-rnn"
        args.write_scores = True
        args.lang = "en"
        args.check_loops = True
        args.max_height = 20
        args.max_node_ratio = 10.
        args.write_scores = False
        args.check_loops = False
        args.normalize = False
        args.time_out = 30
        args.action_stats = False
        self.args = args
        self.num_feature_tokens = args.shapes.max_buffer_size+args.shapes.max_stack_size
        self.logits = logits,
        (
            self.form_indices,
            self.dep_types,
            self.head_indices,
            self.pos,
            self.child_indices,
            self.ner,
            self.height,
            self.inc,
            self.out,
            self.history,
            self.elmo,
            self.sentence_lengths,
            self.history_lengths,
            self.action_counts,
            self.action_ratios,
            self.node_ratios,
            self.root,
        ) = inputs
        self.passages = passages
        args.parser_batch_size = min(512,len(passages))
        self._summary_writer = summary_writer

    def after_create_session(self, session, coord):
        self.session = session
        from tuepa.util import load_numberer_from_file
        import tuepa
        import os

        from evaluate_elmo import ElmoPredictionData, PredictionWrapper
        args = self.args
        # restore numberers
        with open(os.path.join(args.save_dir, tuepa.util.config.LABELS_FILENAME), "r", encoding="utf-8") as file:
            label_numberer = load_numberer_from_file(file)

        with open(os.path.join(args.save_dir, tuepa.util.config.EDGE_FILENAME), "r", encoding="utf-8") as file:
            edge_numberer = load_numberer_from_file(file)

        with open(os.path.join(args.save_dir, tuepa.util.config.DEP_FILENAME), "r", encoding="utf-8") as file:
            dep_numberer = load_numberer_from_file(file)

        with open(os.path.join(args.save_dir, tuepa.util.config.POS_FILENAME), "r", encoding="utf-8") as file:
            pos_numberer = load_numberer_from_file(file)

        with open(os.path.join(args.save_dir, tuepa.util.config.NER_FILENAME), "r", encoding="utf-8") as file:
            ner_numberer = load_numberer_from_file(file)

        args.num_edges = edge_numberer.max
        args.prediction_data = ElmoPredictionData(
            label_numberer=label_numberer,
            pos_numberer=pos_numberer,
            dep_numberer=dep_numberer,
            edge_numberer=edge_numberer,
            ner_numberer=ner_numberer,
        )
        args.num_ner = ner_numberer.max
        res = list(parser.evaluate(self, args, random.sample(self.passages, args.parser_batch_size), train_eval=True))[0]
        values = []
        for k,v in zip(res.titles(),res.fields()):
            values.append(tf.Summary.Value(tag=k, simple_value=float(v)))

        summary = tf.Summary(value=values)

        self._summary_writer.add_summary(summary,tf.train.get_global_step().eval(session=session))

    def score(self, features):
        return self.session.run(self.logits[0], feed_dict={
                    self.form_indices: features['form_indices'],
                    self.dep_types: features['deps'],
                    self.pos: features['pos'],
                    self.child_indices: features['child_indices'],
                    self.ner: features['ner'],
                    self.head_indices: features['heads'],
                    self.height: features['height'],
                    self.inc: features['inc'],
                    self.out: features['out'],
                    self.history: features['history'],
                    self.sentence_lengths: features['sent_lens'],
                    self.history_lengths: features['hist_lens'],
                    self.elmo: features['elmo'],
                    self.node_ratios: features['node_ratios'],
                    self.action_ratios: features['action_ratios'],
                    self.action_counts: features['action_counts'],
                    self.root: features['root']
                })

class PerClassHook(tf.train.SessionRunHook):
    """
    Taken from: https://github.com/tensorflow/tensorboard/issues/227
    Saves a confusion matrix as a Summary so that it can be shown in tensorboard
    """

    def __init__(self, labels, tensor_name, summary_writer):
        """Initializes a `SaveConfusionMatrixHook`.

        :param labels: Iterable of String containing the labels to print for each
                       row/column in the confusion matrix.
        :param confusion_matrix_tensor_name: The name of the tensor containing the confusion
                                             matrix
        :param summary_writer: The summary writer that will save the summary
        """
        self.tensor_name = tensor_name
        self.labels = labels
        self._summary_writer = summary_writer

    def end(self, session):
        m = tf.get_default_graph().get_tensor_by_name(
            self.tensor_name + ':0').eval(session=session).astype(float)
        globalStep = tf.train.get_global_step().eval(session=session)
        figure = self._plot_matrix(m)
        summary = self._figure_to_summary(figure)
        self._summary_writer.add_summary(summary, globalStep)

    def _figure_to_summary(self, fig):
        """
        Converts a matplotlib figure ``fig`` into a TensorFlow Summary object
        that can be directly fed into ``Summary.FileWriter``.
        :param fig: A ``matplotlib.figure.Figure`` object.
        :return: A TensorFlow ``Summary`` protobuf object containing the plot image
                 as a image summary.
        """

        # attach a new canvas if not exists
        if fig.canvas is None:
            matplotlib.backends.backend_agg.FigureCanvasAgg(fig)

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()

        # get PNG data from the figure
        png_buffer = io.BytesIO()
        fig.canvas.print_png(png_buffer)
        png_encoded = png_buffer.getvalue()
        png_buffer.close()

        summary_image = tf.Summary.Image(height=h, width=w, colorspace=4,  # RGB-A
                                         encoded_image_string=png_encoded)
        summary = tf.Summary(value=[tf.Summary.Value(tag=self.tensor_name, image=summary_image)])
        return summary

    def _plot_matrix(self, cm):
        '''
        :param cm: A confusion matrix: A square ```numpy array``` of the same size as self.labels
    `   :return:  A ``matplotlib.figure.Figure`` object with a numerical and graphical representation of the cm array
        '''
        from matplotlib import rcParams
        rcParams.update({'figure.autolayout': True})
        fig = plt.Figure(figsize=(10, 5), dpi=100, facecolor='w', edgecolor='k')
        ax = fig.add_subplot(1, 1, 1)
        classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x) for x in self.labels]
        # classes = ['\n'.join(textwrap.wrap(l, 20)) for l in classes]
        ax.bar(x=classes, height=cm)
        tick_marks = np.arange(len(classes))
        ax.set_xlabel('Class')
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(classes, rotation=-70, ha='left')
        ax.xaxis.set_label_position('bottom')
        ax.xaxis.tick_bottom()

        fig.set_tight_layout(True)
        return fig
