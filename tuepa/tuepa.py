from functools import partial
import json
from glob import glob
from timeit import default_timer
import sys

import numpy as np
import tensorflow as tf
from semstr.convert import FROM_FORMAT, from_text
from ucca import ioutil
import torch
from elmoformanylangs import Embedder
import finalfrontier

from numberer import Numberer
from oracle import Oracle
from states.state import State
from action import Actions
from config import create_argument_parser
from model import ElModel, feed_forward_from_json
import progress
import random

from collections import namedtuple

Data = namedtuple("Data",
                  "stack_and_buffer_features labels history_features "
                  "elmo_embeddings sentence_lengths history_lengths state2sent_index "
                  "shapes ")

Batch = namedtuple("Batch", "stack_and_buffer_features labels history_features "
                            "elmo_embeddings sentence_lengths history_lengths "
                            "padding non_terminals")


# Marks input passages as text so that we don't accidentally train on them
def from_text_format(*args, **kwargs):
    for passage in from_text(*args, **kwargs):
        passage.extra["format"] = "text"
        yield passage


CONVERTERS = {k: partial(c, annotate=True) for k, c in FROM_FORMAT.items()}
CONVERTERS[""] = CONVERTERS["txt"] = from_text_format


def read_passages(files, language="en"):
    expanded = [f for pattern in files for f in sorted(glob(pattern)) or (pattern,)]
    return ioutil.read_files_and_dirs(expanded, sentences=True, paragraphs=False,
                                      converters=CONVERTERS, lang=language)


def extract_features(state, label_numberer, train=True):
    stack = state.stack
    buffer = state.buffer
    stack_features = [node.text if node.text != None else "<NT>" for node in stack]
    buffer_features = [node.text if node.text != None else "<NT>" for node in buffer]
    history_features = [label_numberer.number(str(action), train=train) for action in state.actions]
    return stack_features, buffer_features, history_features


def run_iteration(model, data, batch_size, epoch_steps, train, verbose=False,
                  embedder=None):
    num_batches = epoch_steps
    # perfect_fit = num_batches == (features.shape[0] / batch_size)

    iteration_entropy = 0
    iteration_accuracy = 0
    state2pid = np.array(data.state2sent_index)

    keys = list(range(len(state2pid)))
    for batch_offset in range(1, epoch_steps):
        getters = random.sample(keys, batch_size)

        # lookup associated sentence for each state
        ids = state2pid[getters]
        batch_elmo = np.array(data.elmo_embeddings)[ids]
        max_length = max(map(len, batch_elmo))
        # loop over sentences and pad them to max length in batch
        batch_elmo = [np.vstack([n, np.zeros(
                                        shape=[max_length - len(n),
                                               n.shape[-1]],
                                    dtype=np.float32)])
                      for n in batch_elmo]

        # [batch, time, D]
        batch_elmo = np.transpose(batch_elmo, [0, 1, 2])

        # Map words on stack and buffer to their embeddings. Mapping them ahead of time means excessive use of memory.
        # TODO: find smart way to batch embeddings ahead of time or use threading to prepare batches during the tf calls
        batch_features = data.stack_and_buffer_features[getters]
        embedding_collector = []
        for f in batch_features:
            vec = []
            for e in f:
                if e == 0:
                    vec.append(300 * [0.])
                elif e == "<NT>":
                    vec.append(300 * [1.])
                else:
                    vec.append(embedder.embedding(e))
            embedding_collector.append(vec)

        batch_features = np.array(embedding_collector, dtype=np.float32)
        del embedding_collector

        padding = np.isclose(batch_features.mean(axis=-1), 0.)
        non_terminals = np.isclose(batch_features.mean(axis=-1), 1.)

        batch = Batch(stack_and_buffer_features=batch_features,
                      labels=data.labels[getters],
                      history_features=data.history_features[getters],
                      elmo_embeddings=batch_elmo,
                      padding=padding,
                      non_terminals=non_terminals,
                      sentence_lengths=data.sentence_lengths[ids],
                      history_lengths=data.history_lengths[getters])

        entropy, accuracy = model.run_step(
            batch,
            train=train
        )

        iteration_entropy += (entropy - iteration_entropy) / batch_offset
        iteration_accuracy += (accuracy - iteration_accuracy) / batch_offset

        if verbose:
            progress.print_network_progress(
                "Training" if train else "Validating",
                batch_offset,
                num_batches,
                entropy,
                iteration_entropy,
                accuracy,
                iteration_accuracy
            )

    if verbose:
        progress.clear_progress()

    return iteration_entropy, iteration_accuracy


class Shapes:
    def __init__(self, max_stack_size, max_buffer_size):
        self.max_stack_size = max_stack_size
        self.max_buffer_size = max_buffer_size


def preprocess_dataset(path, args, elmo_embedder, maximum_feature_size=None, max_features=None, label_numberer=None):
    if maximum_feature_size is not None:
        max_stack_size = maximum_feature_size.max_stack_size
        max_buffer_size = maximum_feature_size.max_buffer_size
    else:
        max_stack_size = max_buffer_size = -1
    max_hist_size = -1

    stack_and_buffer_features = []
    labels = []
    history_features = []
    sentence_lengths = []
    history_lengths = []
    passage_id2sent = []
    state2passage_id = []
    passage_id = 0

    for passage in read_passages([path]):
        sent = [str(n) for n in passage.layer("0").words]
        if len(sent) > args.max_training_length and maximum_feature_size is None:
            continue
        passage_id2sent.append(sent)
        sentence_lengths.append(len(sent))
        state = State(passage, args)
        oracle = Oracle(passage, args)

        while not state.finished:
            state2passage_id.append(passage_id)
            actions = oracle.generate_actions(state=state)
            action = next(actions)

            # Assumes that if maximum_features size is None training data is being processed
            stack_features, buffer_features, state_history = extract_features(
                state,
                label_numberer,
                train=maximum_feature_size is None
            )
            history_lengths.append(len(state_history))

            label = label_numberer.number(str(action), train=maximum_feature_size is None)
            state.transition(action)
            action.apply()
            stack_and_buffer_features.append((stack_features, buffer_features))
            history_features.append(state_history)
            if maximum_feature_size is None:
                max_stack_size = max(len(stack_features), max_stack_size)
                max_buffer_size = max(len(buffer_features), max_buffer_size)
            max_hist_size = max(len(state_history), max_hist_size)
            labels.append(label)

        if max_features is not None and len(stack_and_buffer_features) >= max_features:
            break
        passage_id += 1

    num_examples = len(stack_and_buffer_features)

    # Create feature_matrix from stack and buffer features
    # if non-training data contains stacks or buffers larger than the maximum size in the training data it's truncated
    # to max_stack_size or max_buffer_size respectively
    feature_matrix = np.zeros((num_examples, max_stack_size + max_buffer_size), dtype=np.object)
    history_matrix = np.zeros((num_examples, max_hist_size), dtype=np.int32)
    for index, feature in enumerate(stack_and_buffer_features):
        feature_matrix[index, :min(len(feature[0]), max_stack_size)] = feature[0][:max_stack_size]
        feature_matrix[index, max_stack_size:max_stack_size + len(feature[1])] = feature[1][:max_buffer_size]
        history_matrix[index, :min(len(history_features[index]), max_hist_size)] = history_features[index]
    labels = np.array(labels)
    history_lengths = np.array(history_lengths)
    sentence_lengths = np.array(sentence_lengths)

    # produce contextualized embeddings
    contextualized_embeddings = elmo_embedder.sents2elmo(passage_id2sent)
    torch.cuda.empty_cache()

    # Returns generated maximum feature sizes of training data
    # and only features and labels for validation/testing datasets
    if maximum_feature_size is None:
        return Data(stack_and_buffer_features=feature_matrix,
                    labels=labels,
                    shapes=Shapes(max_stack_size, max_buffer_size),
                    history_features=history_matrix,
                    elmo_embeddings=contextualized_embeddings,
                    sentence_lengths=sentence_lengths,
                    state2sent_index=state2passage_id,
                    history_lengths=history_lengths)

    return Data(stack_and_buffer_features=feature_matrix,
                labels=labels,
                shapes=None,
                history_features=history_matrix,
                elmo_embeddings=contextualized_embeddings,
                sentence_lengths=sentence_lengths,
                state2sent_index=state2passage_id,
                history_lengths=history_lengths)


def preprocess_and_train(args):
    if args.verbose:
        # Print to stderr so it doesn't get piped
        print("Processing passages...", end="\r", file=sys.stderr)
    # Time preprocessing
    processing_start_time = default_timer()

    label_numberer = Numberer()
    # Preprocess training set

    ff_emb = finalfrontier.Model(args.ff_path, mmap=True)
    elmo_embedder = Embedder(args.elmo_path, batch_size=32)

    training_data = preprocess_dataset(
        args.training_path,
        args,
        elmo_embedder=elmo_embedder,
        max_features=args.max_training_features,
        label_numberer=label_numberer
    )

    # Preprocess validation set
    validation_data = preprocess_dataset(
        args.validation_path,
        args,
        maximum_feature_size=training_data.shapes,
        elmo_embedder=elmo_embedder,
        max_features=args.max_validation_features,
        label_numberer=label_numberer
    )
    print(label_numberer)
    # Clear the line before printing over it
    if args.verbose:
        print(end="\033[K", file=sys.stderr)
        print("Processing complete in {:.4f}s".format(default_timer() - processing_start_time))
    if args.log_file:
        args.log_file.write("Processing complete in {:.4f}s\n".format(default_timer() - processing_start_time))

    model = ElModel(
        args,
        args.layers,
        label_numberer.max,
        300
    )
    batch_size = args.batch_size

    iteration_count = 0

    # Train forever
    while True:
        iteration_count += 1
        start_time = default_timer()

        # Training iteration
        training_entropy, training_accuracy = run_iteration(
            model,
            training_data,
            embedder=ff_emb,
            batch_size=batch_size,
            epoch_steps=args.epoch_steps,
            train=True,
            verbose=args.verbose
        )

        # Validation iteration
        validation_entropy, validation_accuracy = run_iteration(
            model,
            validation_data,
            embedder=ff_emb,
            batch_size=batch_size,
            epoch_steps=args.epoch_steps,
            train=False,
            verbose=args.verbose
        )

        if args.verbose:
            progress.print_iteration_info(
                iteration_count,
                training_entropy,
                training_accuracy,
                validation_entropy,
                validation_accuracy,
                start_time,
            )
        if args.log_file:
            progress.print_iteration_info(
                iteration_count,
                training_entropy,
                training_accuracy,
                validation_entropy,
                validation_accuracy,
                start_time,
                args.log_file
            )


def main(args):
    tf.enable_eager_execution()
    argument_parser = create_argument_parser()

    args = argument_parser.parse_args(args)

    if args.log_file:
        with args.log_file:
            # Log commandline arguments
            args.log_file.write("{}\n".format(args))
            args.layers = feed_forward_from_json(json.loads(args.layers))
            preprocess_and_train(args)

    else:
        args.layers = feed_forward_from_json(json.loads(args.layers))
        preprocess_and_train(args)


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
