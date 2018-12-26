from functools import partial
import json
from glob import glob
from timeit import default_timer
import sys

import numpy as np
import tensorflow as tf
from semstr.convert import FROM_FORMAT, from_text
from ucca import ioutil, convert

from numberer import Numberer
from oracle import Oracle
from states.state import State
from config import create_argument_parser
from model import FFModel, TransformerModel, feed_forward_from_json
import progress


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


def word_or_nt(node):
    return node.text if node.text else "<NT>"

def extract_features(state, w_numberer, train=True):
    stack = state.stack
    buffer = state.buffer
    stack_features = [w_numberer.number(word_or_nt(node), train=train) for node in stack]
    buffer_features = [w_numberer.number(word_or_nt(node), train=train) for node in buffer]

    return stack_features, buffer_features


def run_iteration(model, features, labels, batch_size, train, verbose=False):
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

        if verbose:
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

        if verbose:
            progress.print_network_progress(
                "Training" if train else "Validating",
                num_batches + 1,
                num_batches + 1,
                entropy,
                iteration_entropy,
                accuracy,
                iteration_accuracy
            )

    if verbose:
        progress.clear_progress()

    return iteration_entropy, iteration_accuracy



class MaximumFeatureSize:

    def __init__(self, max_stack_size, max_buffer_size, max_passage_length):
        self.max_stack_size = max_stack_size
        self.max_buffer_size = max_buffer_size
        self.max_passage_length = max_passage_length


def preprocess_dataset(path, word_numberer, args, maximum_feature_size=None, max_features=None, include_tokens=False, passage_seperator=None):
    if maximum_feature_size is not None:
        max_stack_size = maximum_feature_size.max_stack_size
        max_buffer_size = maximum_feature_size.max_buffer_size
        max_passage_length = maximum_feature_size.max_passage_length

    else:
        max_stack_size = max_buffer_size = max_passage_length = 0

    features = []
    labels = []

    for passage in read_passages([path]):
        if include_tokens:
            passage_tokens = [
                word_numberer.number(token, train=maximum_feature_size is None)
                for token in convert.to_text(passage, sentences=False)[0].split()
            ]

        state = State(passage, args)
        oracle = Oracle(passage, args)

        if maximum_feature_size is None:
            max_passage_length = max(len(passage_tokens), max_passage_length)

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
            features.append(
                (stack_features, buffer_features, passage_tokens) if include_tokens else (stack_features, buffer_features)
            )

            if maximum_feature_size is None:
                max_stack_size = max(len(stack_features), max_stack_size)
                max_buffer_size = max(len(buffer_features), max_buffer_size)

            labels.append(label)

        if max_features is not None and len(features) >= max_features:
            break


    # Create feature_matrix from stack and buffer features
    # if non-training data contains stacks or buffers larger than the maximum size in the training data it's truncated
    # to max_stack_size or max_buffer_size respectively
    has_seperator = passage_seperator is not None
    seperator_length = int(has_seperator)

    feature_matrix = np.zeros(
        (len(features), max_stack_size + max_buffer_size + max_passage_length + seperator_length),
        dtype=np.int32
    )

    passage_length = 0
    for index, feature in enumerate(features):
        if include_tokens:
            passage_length = min(max_passage_length, len(feature[2]))
            feature_matrix[index, :passage_length] = feature[2][:max_passage_length]
            if has_seperator:
                feature_matrix[index, passage_length] = passage_seperator

        feature_matrix[
            index,
            passage_length + seperator_length:passage_length + seperator_length + min(len(feature[0]), max_stack_size)
        ] = feature[0][:max_stack_size]
        feature_matrix[
            index,
            passage_length + seperator_length + max_stack_size:passage_length + seperator_length + max_stack_size + len(feature[1])
        ] = feature[1][:max_buffer_size]

    labels = np.array(labels)

    # Returns generated maximum feature sizes of training data
    # and only features and labels for validation/testing datasets
    if maximum_feature_size is None:
        return feature_matrix, labels, MaximumFeatureSize(max_stack_size, max_buffer_size, max_passage_length)

    return feature_matrix, labels


NUM_LABELS = 12


def preprocess_and_train(args):
    if args.verbose:
        # Print to stderr so it doesn't get piped
        print("Processing passages...", end="\r", file=sys.stderr)

    # Time preprocessing
    processing_start_time = default_timer()
    # Reserves index 0 for padding (e.g. empty stack or buffer slots) and index 1 for unknown tokens
    if args.model_type == "transformer":
        word_numberer = Numberer(first_elements=["<PAD>", "<SEP>"])
    else:
        word_numberer = Numberer(first_elements=["<PAD>"])

    # Preprocess training set
    training_features, training_labels, maximum_feature_size = preprocess_dataset(
        args.training_path,
        word_numberer,
        args,
        max_features=args.max_training_features,
        include_tokens=args.model_type != "feedforward",
        passage_seperator=word_numberer.number("<SEP>", False) if args.model_type == "transformer" else None
    )

    # Preprocess validation set
    validation_features, validation_labels = preprocess_dataset(
        args.validation_path,
        word_numberer,
        args,
        maximum_feature_size,
        max_features=args.max_validation_features,
        include_tokens=args.model_type != "feedforward",
        passage_seperator=word_numberer.number("<SEP>", False) if args.model_type == "transformer" else None
    )

    # Clear the line before printing over it
    if args.verbose:
        print(end="\033[K", file=sys.stderr)
        print("Processing {} passages complete in {:.4f}s".format(
            training_features.shape[0] + validation_features.shape[0],
            default_timer() - processing_start_time
        ))
    if args.log_file:
        args.log_file.write("Processing complete in {:.4f}s\n".format(default_timer() - processing_start_time))

    if args.model_type == "feedforward":
        model = FFModel(
            word_numberer.max,
            args.embedding_size,
            args.layers,
            NUM_LABELS,
            args.learning_rate,
            args.input_dropout,
            args.layer_dropout
        )
    else:
        model = TransformerModel(
            word_numberer.max,
            NUM_LABELS,
            args
        )

    batch_size = args.batch_size

    iteration_count = 0

    # Train forever
    while True:
        iteration_count += 1
        start_time = default_timer()

        # Training iteration
        training_entropy, training_accuracy = run_iteration(
            model, training_features, training_labels, batch_size, train=True, verbose=args.verbose
        )

        # Validation iteration
        validation_entropy, validation_accuracy = run_iteration(
            model, validation_features, validation_labels, batch_size, train=False, verbose=args.verbose
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
            if args.model_type == "feedforward":
                args.layers = feed_forward_from_json(json.loads(args.layers))
            preprocess_and_train(args)

    else:
        if args.model_type == "feedforward":
            args.layers = feed_forward_from_json(json.loads(args.layers))
        preprocess_and_train(args)


if __name__ == "__main__":
    import sys


    main(sys.argv[1:])
