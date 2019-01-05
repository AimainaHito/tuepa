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
from elmoformanylangs import Embedder
import finalfrontier

from numberer import Numberer, load_numberer_from_file
from config import create_argument_parser, save_args, load_args
from model import ElModel, FFModel, TransformerModel, feed_forward_from_json
from preprocessing import PredictionData, preprocess_dataset, read_passages
import progress
import parser


Batch = namedtuple("Batch", "stack_and_buffer_features labels history_features "
                            "elmo_embeddings sentence_lengths history_lengths "
                            "padding non_terminals")


ARGS_FILENAME = "args.pickle"
DICTIONARY_FILENAME = "dictionary.csv"
LABELS_FILENAME = "labels.csv"


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


def run_elmo_iteration(model, data, batch_size, epoch_steps, train, embedding_size,
                       verbose=False, embedder=None):
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
                    vec.append(embedding_size * [0.])
                elif e == "<NT>":
                    vec.append(embedding_size * [1.])
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
    elif args.model_type == "elmo-rnn":
        word_numberer = finalfrontier.Model(args.ff_path, mmap=True)
        elmo_embedder = Embedder(args.elmo_path, batch_size=32)
    else:
        word_numberer = Numberer(first_elements=["<PAD>"])

    label_numberer = Numberer()
    # Preprocess training set
    training_data = preprocess_dataset(
        args.training_path,
        args,
        embedder=elmo_embedder if args.model_type == "elmo-rnn" else word_numberer,
        max_features=args.max_training_features,
        label_numberer=label_numberer,
        passage_seperator=word_numberer.number("<SEP>", False) if args.model_type == "transformer" else None,
        use_elmo=args.model_type == "elmo-rnn"
    )

    # Preprocess validation set
    validation_data = preprocess_dataset(
        args.validation_path,
        args,
        maximum_feature_size=training_data.shapes,
        embedder=elmo_embedder if args.model_type == "elmo-rnn" else word_numberer,
        max_features=args.max_validation_features,
        label_numberer=label_numberer,
        passage_seperator=word_numberer.number("<SEP>", False) if args.model_type == "transformer" else None,
        use_elmo=args.model_type == "elmo-rnn",
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
        if args.model_type != "elmo-rnn":
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

        # Training iteration
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
    if args.model_type != "elmo-rnn":
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
    elif args.model_type == "elmo-rnn":
        model = ElModel(
            args,
            args.layers,
            args.num_labels
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
