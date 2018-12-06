from functools import partial
import json
from glob import glob
from timeit import default_timer

import numpy as np
import tensorflow as tf
from semstr.convert import FROM_FORMAT, from_text
from ucca import ioutil

from numberer import Numberer
from oracle import Oracle
from states.state import State
from config import create_argument_parser
from model import FFModel, feed_forward_from_json
import progress


# Marks input passages as text so that we don't accidentally train on them
def from_text_format(*args, **kwargs):
    for passage in from_text(*args, **kwargs):
        passage.extra["format"] = "text"
        yield passage


CONVERTERS = {k: partial(c, annotate=True) for k, c in FROM_FORMAT.items()}
CONVERTERS[""] = CONVERTERS["txt"] = from_text_format


def read_passages(files):
    expanded = [f for pattern in files for f in sorted(glob(pattern)) or (pattern,)]
    return ioutil.read_files_and_dirs(expanded, sentences=True, paragraphs=False,
                                      converters=CONVERTERS, lang="en")


def extract_features(state, w_numberer, train=True):
    stack = state.stack
    buffer = state.buffer
    stack_features = [w_numberer.number(node.text, train=train) for node in stack]
    buffer_features = [w_numberer.number(node.text, train=train) for node in buffer]

    return stack_features, buffer_features


def run_iteration(model, features, labels, batch_size, train):
    num_batches = features.shape[0] // batch_size
    perfect_fit = num_batches == (features.shape[0] / batch_size)

    iteration_entropy = 0
    iteration_accuracy = 0

    for batch_offset in range(1, num_batches + 1):
        entropy, accuracy = model.run_step(
            features[
                (batch_offset - 1) * batch_size:batch_offset * batch_size
            ],
            labels[(batch_offset - 1) * batch_size:batch_offset * batch_size],
            train=train
        )

        iteration_entropy += (entropy - iteration_entropy) / batch_offset
        iteration_accuracy += (accuracy - iteration_accuracy) / batch_offset

        progress.print_network_progress(
            "Training" if train else "Validating",
            batch_offset,
            num_batches + int(not perfect_fit),
            entropy,
            iteration_entropy,
            accuracy,
            iteration_accuracy
        )

    # Handle remaining data
    if not perfect_fit:
        entropy, accuracy = model.run_step(
            features[num_batches * batch_size:],
            labels[num_batches * batch_size:],
            train=train
        )

        iteration_entropy += (entropy - iteration_entropy) / (num_batches + 1)
        iteration_accuracy += (accuracy - iteration_accuracy) / (num_batches + 1)

        progress.print_network_progress(
            "Training" if train else "Validating",
            num_batches + 1,
            num_batches + 1,
            entropy,
            iteration_entropy,
            accuracy,
            iteration_accuracy
        )

    progress.clear_progress()

    return iteration_entropy, iteration_accuracy



class MaximumFeatureSize:

    def __init__(self, max_stack_size, max_buffer_size):
        self.max_stack_size = max_stack_size
        self.max_buffer_size = max_buffer_size


def preprocess_dataset(path, word_numberer, args, maximum_feature_size=None, max_features=None):
    if maximum_feature_size is not None:
        max_stack_size = maximum_feature_size.max_stack_size
        max_buffer_size = maximum_feature_size.max_buffer_size
    else:
        max_stack_size = max_buffer_size = -1

    features = []
    labels = []

    for passage in read_passages([path]):
        state = State(passage, args)
        oracle = Oracle(passage, args)

        while not state.finished:
            actions = oracle.generate_actions(state=state)
            action = next(actions)
            # Assumes that if maximum_features size is None training data is being processed
            stack_features, buffer_features = extract_features(
                state,
                word_numberer,
                train=maximum_feature_size is None
            )
            label = action.type_id
            state.transition(action)
            action.apply()
            features.append((stack_features, buffer_features))

            if maximum_feature_size is None:
                max_stack_size = max(len(stack_features), max_stack_size)
                max_buffer_size = max(len(buffer_features), max_buffer_size)

            labels.append(label)

        if max_features is not None and len(features) >= max_features:
            break

    feature_matrix = np.zeros((len(features), max_stack_size + max_buffer_size), dtype=np.int32)
    for index, feature in enumerate(features):
        feature_matrix[index, :min(len(feature[0]), max_stack_size)] = feature[0][:max_stack_size]
        feature_matrix[index, max_stack_size:max_stack_size + len(feature[1])] = feature[1][:max_buffer_size]

    labels = np.array(labels)

    if maximum_feature_size is None:
        return feature_matrix, labels, MaximumFeatureSize(max_stack_size, max_buffer_size)

    return feature_matrix, labels


def main(args):
    tf.enable_eager_execution()
    argument_parser = create_argument_parser()

    args = argument_parser.parse_args(args)
    args.layers = feed_forward_from_json(json.loads(args.layers))

    word_numberer = Numberer(first_elements=["<PAD>"])

    print("Processing passages...", end="\r")
    processing_start_time = default_timer()

    training_features, training_labels, maximum_feature_size = preprocess_dataset(
        args.training_path,
        word_numberer,
        args,
        max_features=args.max_training_features
    )

    validation_features, validation_labels = preprocess_dataset(
        args.validation_path,
        word_numberer,
        args,
        maximum_feature_size,
        max_features=args.max_validation_features
    )

    print("\033[KProcessing complete in {:.4f}s".format(default_timer() - processing_start_time))

    model = FFModel(word_numberer.max, args.embedding_size, args.layers)
    batch_size = args.batch_size

    iteration_count = 0

    while True:
        iteration_count += 1
        start_time = default_timer()

        training_entropy, training_accuracy = run_iteration(
            model, training_features, training_labels, batch_size, train=True
        )

        validation_entropy, validation_accuracy = run_iteration(
            model, validation_features, validation_labels, batch_size, train=False
        )

        progress.print_iteration_info(
            iteration_count,
            training_entropy,
            training_accuracy,
            validation_entropy,
            validation_accuracy,
            start_time
        )
        '''
        print("Iteration {} | Entropy: {:.2f}, Accuracy: {:.2%}".format(
            iteration_count,
            # Turn np.float into a numpy float for better formatting
            training_entropy,
            training_accuracy
        ))
        '''


if __name__ == "__main__":
    import sys


    main(sys.argv[1:])
