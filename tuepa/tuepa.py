from functools import partial
import json
from glob import glob

import numpy as np
import tensorflow as tf
from semstr.convert import FROM_FORMAT, from_text
from ucca import ioutil

from numberer import Numberer
from oracle import Oracle
from states.state import State
from config import create_argument_parser
from model import FFModel, feed_forward_from_json


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
    stack_features = [w_numberer.number(e, train=train) for e in stack]
    buffer_features = [w_numberer.number(e, train=train) for e in buffer]

    return stack_features, buffer_features


def main(args):
    tf.enable_eager_execution()
    argument_parser = create_argument_parser()

    args = argument_parser.parse_args(args)
    args.layers = feed_forward_from_json(json.loads(args.layers))

    word_numberer = Numberer(first_elements=["<PAD>"])
    # nt_numberer = Numberer()

    features = []
    labels = []

    max_stack_size = max_buffer_size = -1
    print("Processing passages...", end="\r")
    for passage in read_passages([args.path]):
        state = State(passage, args)
        oracle = Oracle(passage, args)

        while not state.finished:
            actions = oracle.generate_actions(state=state)
            action = next(actions)
            stack_features, buffer_features = extract_features(state, word_numberer)
            label = action.type_id
            state.transition(action)
            action.apply()
            features.append((stack_features, buffer_features))

            max_stack_size = max(len(stack_features), max_stack_size)
            max_buffer_size = max(len(buffer_features), max_buffer_size)

            labels.append(label)

        if args.max_features is not None and len(features) > args.max_features:
            break

    print("\033[KProcessing complete")
    feature_matrix = np.zeros((len(features), max_stack_size + max_buffer_size), dtype=np.int32)
    for index, feature in enumerate(features):
        feature_matrix[index, :len(feature[0])] = feature[0]
        feature_matrix[index, max_stack_size:max_stack_size + len(feature[1])] = feature[1]

    # Save memory
    del features

    labels = np.array(labels)

    model = FFModel(word_numberer.max, args.embedding_size, args.layers)
    batch_size = args.batch_size

    num_batches = feature_matrix.shape[0] // batch_size
    perfect_fit = num_batches == (feature_matrix.shape[0] / batch_size)
    iteration_count = 0

    while True:
        iteration_count += 1
        batch_entropy = []
        batch_accuracy = []

        for batch_offset in range(num_batches):
            entropy, accuracy = model.run_step(
                feature_matrix[
                    batch_offset * batch_size:(batch_offset + 1) * batch_size
                ],
                labels[batch_offset * batch_size:(batch_offset + 1) * batch_size],
                train=True
            )

            batch_entropy.append(entropy)
            batch_accuracy.append(accuracy)

        # Handle remaining data
        if not perfect_fit:
            entropy, accuracy = model.run_step(
                feature_matrix[num_batches * batch_size:],
                labels[num_batches * batch_size:],
                train=True
            )

            batch_entropy.append(entropy)
            batch_accuracy.append(accuracy)

        print("Iteration {} | Entropy: {:.2f}, Accuracy: {:.2%}".format(
            iteration_count,
            # Turn np.float into a numpy float for better formatting
            np.array(batch_entropy).mean(),
            np.array(accuracy).mean()
        ))

if __name__ == "__main__":
    import sys


    main(sys.argv[1:])
