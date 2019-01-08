import sys
from timeit import default_timer
from collections import defaultdict
import concurrent.futures
import os
from enum import Enum
from functools import partial
from glob import glob

import numpy as np
from semstr.convert import TO_FORMAT
from semstr.evaluate import EVALUATORS, Scores
from ucca import ioutil
from ucca.evaluation import LABELED, UNLABELED, EVAL_TYPES, evaluate as evaluate_ucca
from ucca.normalization import normalize

from .states.state import State
from .action import Action
import tuepa.data.preprocessing as preprocessing
import tuepa.data.elmo.elmo_processing as preprocess_elmo

class ParserException(Exception):
    pass


class ParseMode(Enum):
    train = 1
    dev = 2
    test = 3


class AbstractParser:
    def __init__(self, models, argus, evaluation=False):
        self.models = models
        self.evaluation = evaluation
        self.args = argus
        self.action_count = self.correct_action_count = self.label_count = 0
        self.correct_label_count = self.num_tokens = self.f1 = 0
        self.started = default_timer()

    @property
    def model(self):
        return self.models[0]

    @model.setter
    def model(self, model):
        self.models[0] = model

    @property
    def duration(self):
        return (default_timer() - self.started) or 1.0

    def tokens_per_second(self):
        return self.num_tokens / self.duration


class PassageParser(AbstractParser):
    """ Parser for a single passage, has a state and optionally an oracle """

    def __init__(self, passage, args, models, evaluation, **kwargs):
        super().__init__(models, args, evaluation)
        self.passage = self.out = passage
        if self.args.model_type != "feedforward":
            self.sentence_tokens = [str(n) for n in passage.layer("0").words]

        self.format = self.passage.extra.get("format")

        self.in_format = self.format or "ucca"
        self.out_format = "ucca" if self.format in (None, "text") else self.format
        self.lang = self.passage.attrib.get("lang", self.args.lang)

        # Used in verify_passage to optionally ignore a mismatch in linkage nodes:
        self.state_hash_history = set()
        self.state = self.eval_type = None

    def init(self):
        self.state = State(self.passage, self.args)

    def parse(self, display=True, write=False):
        self.init()
        passage_id = self.passage.ID

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                executor.submit(self.parse_internal).result(self.args.timeout)
            status = "(%d tokens/s)" % self.tokens_per_second()
        except ParserException as e:
            print("%s: %s" % (passage_id, e), file=sys.stderr)
            status = "(failed)"
        except concurrent.futures.TimeoutError:
            print("%s: timeout (%fs)" % (passage_id, self.args.timeout), file=sys.stderr)
            status = "(timeout)"

        return self.finish(status, display=display, write=write)

    def parse_internal(self):
        """
        Internal method to parse a single passage with the classifier.
        """
        # print("  initial state: %s" % self.state)

        while True:
            if self.args.check_loops:
                self.check_loop()

            # self.label_node()  # In case root node needs labeling
            action = self.choose()
            self.state.transition(action)
            print(self.state.stack)
            print(self.state.buffer)
            if self.args.action_stats:
                try:
                    if self.args.action_stats == "-":
                        print(action)
                    else:
                        with open(self.args.action_stats, "a") as file:
                            file.write("{}\n".format(action))

                except OSError:
                    pass

            # print("\n".join(["  predicted: %-15s true: %-15s taken: %-15s %s" % (
            #    predicted_action, "|".join(map(str, true_actions.values())), action, self.state) if self.oracle else
            #                              "  action: %-15s %s" % (action, self.state)] + (
            #    ["  predicted label: %-9s true label: %s" % (predicted_label, true_label) if self.oracle and not
            #     self.args.use_gold_node_labels else "  label: %s" % label] if need_label else []) + [
            #    "    " + l for l in self.state.log]))

            if self.state.finished:
                return  # action is Finish (or early update is triggered)

    def choose(self, name="action"):
        # if axis == NODE_LABEL_KEY:
        #    is_valid = self.state.is_valid_label
        # else:
        is_valid = self.state.is_valid_action
        max_stack_size = self.args.shapes.max_stack_size
        max_buffer_size = self.args.shapes.max_buffer_size

        if self.args.model_type != "elmo-rnn":
            stack_features, buffer_features, _ = preprocessing.extract_numbered_features(
                self.state,
                self.args.prediction_data.embedder,
                self.args.prediction_data.label_numberer,
                train=False
            )

            if self.args.model_type == "feedforward":
                features = np.zeros(
                    (1, max_stack_size + max_buffer_size),
                    dtype=np.int32
                )

                preprocessing.add_stack_and_buffer_features(
                    features,
                    0,
                    stack_features,
                    buffer_features,
                    max_stack_size,
                    max_buffer_size
                )

            else:
                features = np.zeros(
                    (1, max_stack_size + max_buffer_size + self.args.max_training_length + 1),
                    dtype=np.int32
                )

                preprocessing.add_transformer_features(
                    features,
                    0,
                    self.sentence_tokens,
                    stack_features,
                    buffer_features,
                    "<SEP>",
                    self.args.max_training_length,
                    max_stack_size,
                    max_buffer_size
                )
            scores, = self.models[0].score(features).numpy()
        else:
            stack_features, buffer_features, history_features = preprocess_elmo.extract_elmo_features(self.args,self.state,
                                                                                               self.args.prediction_data.label_numberer,
                                                                                               self.args.prediction_data.pos_numberer,
                                                                                               self.args.prediction_data.dep_numberer,
                                                                                               self.args.prediction_data.edge_numberer,
                                                                                               train=False)
            forms, deps, heads, pos, incoming, outgoing, height = tuple(zip(*(stack_features + buffer_features)))

            actions = [self.args.prediction_data.label_numberer.value2num[str(action)] for action in self.state.actions]
            previous_actions = np.zeros((self.args.prediction_data.label_numberer.max),dtype=np.int32)
            previous_actions[actions] += 1

            hist_len = [len(history_features)]

            if not history_features:
                history_features += [0]

            inc = np.zeros((max_stack_size+max_buffer_size,self.args.num_edges),dtype=np.int32)
            out = np.zeros((max_stack_size+max_buffer_size,self.args.num_edges),dtype=np.int32)
            for n, item in enumerate(incoming):
                for id in item:
                    inc[n, id] += 1
            for n, item in enumerate(outgoing):
                for id in item:
                    out[n, id] += 1

            elmo = self.state.passage.elmo[0]
            sent_length = len(elmo)
            print(history_features)
            # import IPython; IPython.embed()
            features = {'form_indices': np.array([forms],dtype=np.int32),
                        'deps': np.array([deps],dtype=np.int32),
                        'pos': np.array([pos],dtype=np.int32),
                        'heads':np.array([heads], dtype=np.int32),
                        'height': np.array([height],dtype=np.int32),
                        'inc' : np.array([inc],dtype=np.int32),
                        'out': np.array([out], dtype=np.int32),
                        'elmo': np.array([elmo], np.float32),
                        'sent_lens': np.array([sent_length], np.int32),
                        'history': np.array([history_features], dtype=np.int32),
                        'hist_lens': np.array(hist_len, np.int32),
                        'action_counts': previous_actions.reshape([1,-1])}

            scores, = self.models[0].score(features)
            print(scores.argmax)
            # import IPython; IPython.embed()
            # TODO: Implement elmo-rnn prediction

        # for model in self.models[1:]:  # Ensemble if given more than one model; align label order and add scores
        #    label_scores = dict(zip(model.classifier.labels[axis].all, self.model.score(self.state, axis)[0]))
        #    scores += [label_scores.get(a, 0) for a in labels.all]  # Product of Experts, assuming log(softmax)
        try:
            print(str(self.predict(scores, is_valid)))
            return self.predict(scores, is_valid)
        except StopIteration as e:
            raise ParserException("No valid %s available\n" % name) from e

    def predict(self, scores, is_valid=None):
        """
        Choose action/label based on classifier
        Usually the best action/label is valid, so max is enough to choose it in O(n) time
        Otherwise, sorts all the other scores to choose the best valid one in O(n lg n)
        :return: valid action/label with maximum probability according to classifier
        """
        for i in scores.argsort()[::-1]:
            action = self.args.prediction_data.label_numberer.value(i)
            split_act = action.split("-")
            tag = None
            if len(split_act) > 1:
                action = "-".join(split_act[:-1])
                tag = split_act[-1]
            action = Action(action,tag=tag)
            if is_valid(action):
                return action

    def finish(self, status, display=True, write=False):
        self.out = self.state.create_passage(verify=self.args.verify, format=self.out_format)
        if write:
            for out_format in self.args.formats or [self.out_format]:
                if self.args.normalize and out_format == "ucca":
                    normalize(self.out)

                ioutil.write_passage(self.out, output_format=out_format, binary=out_format == "pickle",
                                     outdir=self.args.outdir, prefix=self.args.prefix,
                                     converter=get_output_converter(out_format), verbose=self.args.verbose,
                                     append=self.args.join, basename=self.args.join)

        ret = (self.out,)
        if self.evaluation:
            ret += (self.evaluate(),)
            status = "%-14s %s F1=%.3f" % (status, self.eval_type, self.f1)
        if display:
            print("%.3fs %s" % (self.duration, status))
        return ret

    def evaluate(self):
        if self.format:
            print("Converting to %s and evaluating..." % self.format)
        self.eval_type = LABELED
        evaluator = EVALUATORS.get(self.format, evaluate_ucca)
        score = evaluator(self.out, self.passage, converter=get_output_converter(self.format),
                          verbose=self.out,
                          constructions=self.args.constructions,
                          eval_types=(LABELED, UNLABELED))
        self.f1 = average_f1(score, self.eval_type)
        score.lang = self.lang
        return score

    def check_loop(self):
        """
        Check if the current state has already occurred, indicating a loop
        """
        state_hash = hash(self.state)

        assert state_hash not in self.state_hash_history, \
            "\n".join(["Transition loop", self.state.str("\n")])

        self.state_hash_history.add(state_hash)

    @property
    def num_tokens(self):
        return len(set(self.state.terminals).difference(self.state.buffer))  # To count even incomplete parses

    @num_tokens.setter
    def num_tokens(self, _):
        pass


class BatchParser(AbstractParser):
    """ Parser for a single pass over dev/test passages """

    def __init__(self, argus, models, evaluate, **kwargs):
        super().__init__(argus=argus, models=models, evaluation=evaluate, **kwargs)
        self.seen_per_format = defaultdict(int)
        self.num_passages = 0
        from elmoformanylangs import Embedder

        self.elmo = Embedder(argus.elmo_path, batch_size=1)

    def parse(self, passages, display=True, write=False):
        passages, total = generate_and_len(single_to_iter(passages))
        pr_width = len(str(total))
        id_width = 1
        import torch
        for i, passage in enumerate(passages, start=1):
            passage.set_elmo(self.elmo.sents2elmo([[str(n) for n in passage.layer("0").all]]))
            torch.cuda.empty_cache()
            parser = PassageParser(passage, args=self.args, models=self.models, evaluation=self.evaluation)

            if display:
                progress = "%3d%% %*d/%d" % (i / total * 100, pr_width, i, total) if total and i <= total else "%d" % i
                id_width = max(id_width, len(str(passage.ID)))
                print("%s %2s %-6s %-*s" % (progress, parser.lang, parser.in_format, id_width, passage.ID), end="\r")

            self.seen_per_format[parser.in_format] += 1
            yield parser.parse(display=display, write=write)
            self.update_counts(parser)

        if (self.num_passages > 0) and display:
            self.summary()

    def update_counts(self, parser):
        self.correct_action_count += parser.correct_action_count
        self.action_count += parser.action_count
        self.correct_label_count += parser.correct_label_count
        self.label_count += parser.label_count
        self.num_tokens += parser.num_tokens
        self.num_passages += 1
        self.f1 += parser.f1

    def summary(self):
        print("Parsed {} passage{}".format(self.num_passages, "s" if self.num_passages != 1 else ""))
        if self.correct_action_count:
            accuracy_str = percents_str(self.correct_action_count, self.action_count, "correct actions ")
            if self.label_count:
                accuracy_str += ", " + percents_str(self.correct_label_count, self.label_count, "correct labels ")
            print("Overall %s" % accuracy_str)

        print(
            "Total time: %.3fs (average time/passage: %.3fs, average tokens/s: %d)" % (
                self.duration, self.time_per_passage(), self.tokens_per_second()
            ),
            flush=True
        )

    def time_per_passage(self):
        return self.duration / self.num_passages


class Parser(AbstractParser):
    """ Main class to implement transition-based UCCA parser """

    def __init__(self, models, args):
        super().__init__(models, args)
        self.iteration = self.epoch = self.batch = None

    def eval(self, passages, mode, scores_filename, display=True):
        print("Evaluating on %s passages" % mode.name)
        passage_scores = [s for _, s in self.parse(passages, evaluate=True, display=display)]
        scores = Scores(passage_scores)
        average_score = average_f1(scores)
        prefix = ".".join(map(str, [self.iteration, self.epoch]))

        if display:
            print("Evaluation %s, average %s F1 score on %s: %.3f%s" % (
                prefix, LABELED, mode.name,
                average_score, scores.details(average_f1)
            ))

        print_scores(scores, scores_filename, prefix=prefix, prefix_title="iteration")
        return average_score, scores

    def parse(self, passages, evaluate=False, display=True, write=False):
        """
        Parse given passages
        :param passages: iterable of passages to parse
        :param evaluate: whether to evaluate parsed passages with respect to given ones.
                           Only possible when given passages are annotated.
        :param display: whether to display information on each parsed passage
        :param write: whether to write output passages to file
        :return: generator of parsed passages
                 or pairs of (Passage, Scores) if evaluate is set to `True`
        """
        self.batch = 0
        parser = BatchParser(self.args, self.models, evaluate)
        yield from parser.parse(passages, display=display, write=write)


def evaluate(model, args, test_passages):
    """
    Evaluate parse trees generated by the model on the given passages

    :param test_passages: passages to test on
    :param args: evaluation arguments

    :return: generator of test Scores objects
    """
    parser = Parser(models=[model], args=args)

    passage_scores = []
    for result in parser.parse(test_passages, evaluate=True, write=args.write_scores):
        _, *score = result
        passage_scores += score

    if passage_scores:
        scores = Scores(passage_scores)
        print("\nAverage %s F1 score on test: %.3f" % (LABELED, average_f1(scores)))
        print("Aggregated scores:")
        scores.print()
        print_scores(scores, args.log_file if args.log_file else sys.stdout)
        yield scores


def get_output_converter(out_format, default=None, wikification=False):
    converter = TO_FORMAT.get(out_format)
    return partial(converter, wikification=wikification,
                   verbose=True) if converter else default


def percents_str(part, total, infix="", fraction=True):
    ret = "%d%%" % (100 * part / total)
    if fraction:
        ret += " %s(%d/%d)" % (infix, part, total)
    return ret


def print_scores(scores, file, prefix=None, prefix_title=None):
    # if print_title:
    titles = scores.titles()
    if prefix_title is not None:
        titles = [prefix_title] + titles
    print(",".join(titles), file=file)

    fields = scores.fields()
    if prefix is not None:
        fields.insert(0, prefix)
    print(",".join(fields), file=file)


def single_to_iter(it):
    return it if hasattr(it, "__iter__") else (it,)  # Single passage given


def generate_and_len(it):
    return it, (len(it) if hasattr(it, "__len__") else None)


def average_f1(scores, eval_type=None):
    for element in (eval_type or LABELED,) + EVAL_TYPES:
        try:
            return scores.average_f1(element)
        except ValueError:
            pass
    return 0
