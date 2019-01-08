import json
import pickle
from timeit import default_timer
import sys
from collections import namedtuple
from argparse import Namespace
import random
import os
import copy

import numpy as np
import tensorflow as tf

from tuepa.numberer import Numberer, load_numberer_from_file
from tuepa.config import create_argument_parser
from tuepa.model import FFModel, TransformerModel, feed_forward_from_json
from tuepa.preprocessing import PredictionData, preprocess_dataset, read_passages
import tuepa.progress
import parser


Batch = namedtuple("Batch", "stack_and_buffer_features labels history_features "
                            "elmo_embeddings sentence_lengths history_lengths "
                            "padding non_terminals")


ARGS_FILENAME = "args.pickle"
DICTIONARY_FILENAME = "dictionary.csv"
LABELS_FILENAME = "labels.csv"


def save_args(args, model_dir):
    with open(os.path.join(model_dir, ARGS_FILENAME), "wb") as file:
        # Remove log file and tensorflow layers since they can't be serialized
        args = copy.copy(args)
        if "log_file" in args:
            del args.log_file
        if "json_layers" in args:
            del args.layers

        pickle.dump(args, file)

def load_args(model_dir):
    with open(os.path.join(model_dir, ARGS_FILENAME), "rb") as file:
        return pickle.load(file)


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



def preprocess_and_train(args):
    # Create destination directory for saved models
    if args.save_dir:
        os.makedirs(args.save_dir)

    if args.verbose:
        # Print to stderr so it doesn't get piped
        print("Processing passages...", end="\r", file=sys.stderr)

    # Time preprocessing
    processing_start_time = default_timer()

    if args.model_type == "transformer":
        word_numberer = Numberer(first_elements=["<PAD>", "<SEP>"])
    else:
        word_numberer = Numberer(first_elements=["<PAD>"])

    label_numberer = Numberer()
    # Preprocess training set
    training_data = preprocess_dataset(
        args.training_path,
        args,
        embedder=word_numberer,
        max_features=args.max_training_features,
        label_numberer=label_numberer,
        passage_seperator=word_numberer.number("<SEP>", False) if args.model_type == "transformer" else None,
    )

    # Preprocess validation set
    validation_data = preprocess_dataset(
        args.validation_path,
        args,
        maximum_feature_size=training_data.shapes,
        embedder=word_numberer,
        max_features=args.max_validation_features,
        label_numberer=label_numberer,
        passage_seperator=word_numberer.number("<SEP>", False) if args.model_type == "transformer" else None,
    )
    # Clear the line before printing over it
    if args.verbose:
        print(end="\033[K", file=sys.stderr)
        print("Processing {} passages complete in {:.4f}s".format(
            len(training_data.sentence_lengths) + len(validation_data.sentence_lengths),
            default_timer() - processing_start_time
        ))

    if args.log_file:
        args.log_file.write("Processing complete in {:.4f}s\n".format(default_timer() - processing_start_time))

    args.num_labels = label_numberer.max
    args.shapes = training_data.shapes
    if args.model_type == "feedforward":
        args.num_words = word_numberer.max

        model = FFModel(
            args.num_words,
            args.embedding_size,
            args.layers,
            args.num_labels,
            args.learning_rate,
            args.input_dropout,
            args.layer_dropout
        )
    else:
        args.num_words = word_numberer.max

        model = TransformerModel(
            args.num_words,
            args.num_labels,
            args
        )

    # Save arguments and dictionaries
    if args.save_dir:
        with open(os.path.join(args.save_dir, DICTIONARY_FILENAME), "w", encoding="utf-8") as file:
            word_numberer.to_file(file)

        with open(os.path.join(args.save_dir, LABELS_FILENAME), "w", encoding="utf-8") as file:
            label_numberer.to_file(file)

        save_args(args, args.save_dir)

    batch_size = args.batch_size

    iteration_count = 0

    # Train forever
    while True:
        iteration_count += 1
        start_time = default_timer()

        training_entropy, training_accuracy = run_iteration(
            model,
            training_data.stack_and_buffer_features,
            training_data.labels,
            batch_size,
            train=True,
            verbose=args.verbose
        )

        # Validation iteration
        validation_entropy, validation_accuracy = run_iteration(
            model,
            validation_data.stack_and_buffer_features,
            validation_data.labels,
            batch_size,
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
        if args.save_dir:
            model.save(os.path.join(args.save_dir, args.model_type.replace("-", "_")))


def evaluate(args):
    model_args = load_args(args.model_dir)
    # Merge model args with evaluation args
    args = Namespace(**{**vars(args), **vars(model_args)})

    # Load dictionaries
    with open(os.path.join(args.save_dir, DICTIONARY_FILENAME), "r", encoding="utf-8") as file:
        word_numberer = load_numberer_from_file(file)

    with open(os.path.join(args.save_dir, LABELS_FILENAME), "r", encoding="utf-8") as file:
        label_numberer = load_numberer_from_file(file)

    if "json_layers" in args:
        args.layers = feed_forward_from_json(json.loads(args.json_layers))

    if args.model_type == "feedforward":
        model = FFModel(
            args.num_words,
            args.embedding_size,
            args.layers,
            args.num_labels,
            args.learning_rate,
            args.input_dropout,
            args.layer_dropout
        )
    else:
        model = TransformerModel(
            args.num_words,
            args.num_labels,
            args
        )

    args.prediction_data = PredictionData(
        label_numberer=label_numberer,
        embedder=word_numberer,
    )

    model.restore(os.path.join(args.model_dir, args.model_type.replace("-", "_")), args)

    print(*parser.evaluate(model, args, read_passages([args.eval_data])), sep="\n")


def main(args):
    tf.enable_eager_execution()
    argument_parser = create_argument_parser()

    args = argument_parser.parse_args(args)

    if args.model_type == 'elmo-rnn':
        print("I can't run elmo-rnn! \ņ"
              "Run preprocess_elmo.py to preprocess training data. \n"
              "Run train_elmo.py to train the elmo-rnn. \n"
              "Run evaluate_elmo.py to evaluate the trained elmo-rnn.")

    if args.model_type == "evaluate":
        evaluate(args)
        return

    if args.log_file:
        with args.log_file:
            # Log commandline arguments
            args.log_file.write("{}\n".format(args))
            if args.model_type != "transformer":
                args.json_layers = args.layers
                args.layers = feed_forward_from_json(json.loads(args.layers))
            preprocess_and_train(args)

    else:
        if args.model_type != "transformer":
            args.json_layers = args.layers
            args.layers = feed_forward_from_json(json.loads(args.layers))
        preprocess_and_train(args)


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
