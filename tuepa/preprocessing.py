from functools import partial
from glob import glob
from collections import namedtuple

import numpy as np
import torch
from semstr.convert import FROM_FORMAT, from_text
from ucca import ioutil

from oracle import Oracle
from states.state import State
from action import Actions


Data = namedtuple("Data",
                  "stack_and_buffer_features labels history_features "
                  "elmo_embeddings sentence_lengths history_lengths state2sent_index "
                  "shapes ")


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
    stack_features = [node.text if node.text is not None else "<NT>" for node in state.stack[::-1]]
    buffer_features = [node.text if node.text is not None else "<NT>" for node in state.buffer]
    history_features = [label_numberer.number(str(action), train=train) for action in state.actions]

    return stack_features, buffer_features, history_features


def number_node(word_numberer, node, train=False):
    return word_numberer.number(node.text if node.text else "<NT>", train=train)


def extract_numbered_features(state, word_numberer, label_numberer, train=True):
    stack_features = [number_node(word_numberer, node, train) for node in state.stack[::-1]]
    buffer_features = [number_node(word_numberer, node, train) for node in state.buffer]
    history_features = [label_numberer.number(str(action), train=train) for action in state.actions]

    return stack_features, buffer_features, history_features


def add_stack_and_buffer_features(feature_matrix, index, stack_features, buffer_features, max_stack_size, max_buffer_size):
    feature_matrix[index, :min(len(stack_features), max_stack_size)] = stack_features[:max_stack_size]
    feature_matrix[index, max_stack_size:max_stack_size + len(buffer_features)] = buffer_features[:max_buffer_size]


def add_history_features(history_matrix, index, history_features, max_hist_size):
    history_matrix[index, :min(len(history_features), max_hist_size)] = history_features


def add_transformer_features(
        feature_matrix, index, sentence_tokens, stack_features, buffer_features,
        sentence_separator, sentence_length, max_stack_size, max_buffer_size
    ):

    feature_matrix[index, :sentence_length] = sentence_tokens

    feature_matrix[index, sentence_length] = sentence_separator

    feature_matrix[
        index,
        sentence_length + 1:sentence_length + 1 + min(len(stack_features), max_stack_size)
    ] = stack_features[:max_stack_size]
    feature_matrix[
        index,
        sentence_length + 1 + max_stack_size:sentence_length + 1 + max_stack_size + len(buffer_features)
    ] = buffer_features[:max_buffer_size]


class Shapes:
    def __init__(self, max_stack_size, max_buffer_size):
        self.max_stack_size = max_stack_size
        self.max_buffer_size = max_buffer_size


def preprocess_dataset(path, args, embedder, maximum_feature_size=None, max_features=None, label_numberer=None, passage_seperator=None, use_elmo=False):
    has_seperator = passage_seperator is not None

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
        if has_seperator or use_elmo:
            sentence = [str(n) for n in passage.layer("0").words]
            if len(sentence) > args.max_training_length and maximum_feature_size is None:
                continue

            passage_id2sent.append(sentence)
            sentence_lengths.append(len(sentence))

        state = State(passage, args)
        oracle = Oracle(passage, args)

        while not state.finished:
            state2passage_id.append(passage_id)
            actions = oracle.generate_actions(state=state)
            action = next(actions)

            # Assumes that if maximum_features size is None training data is being processed
            if use_elmo:
                stack_features, buffer_features, state_history = extract_features(
                    state,
                    label_numberer,
                    train=maximum_feature_size is None
                )
            else:
                stack_features, buffer_features, state_history = extract_numbered_features(
                    state,
                    embedder,
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
    history_matrix = np.zeros((num_examples, max_hist_size), dtype=np.int32)

    # Create feature_matrix from stack and buffer features
    # if non-training data contains stacks or buffers larger than the maximum size in the training data it's truncated
    # to max_stack_size or max_buffer_size respectively
    if has_seperator:
        feature_matrix = np.zeros(
            (num_examples, max_stack_size + max_buffer_size + args.max_training_length + 1),
            dtype=np.int32
        )
        # Preprocess data for the transformer model
        for index, ((stack_features, buffer_features), passage_id) in enumerate(zip(stack_and_buffer_features, state2passage_id)):
            sentence = passage_id2sent[passage_id]
            add_transformer_features(
                feature_matrix,
                index,
                [embedder.number(token, train=maximum_feature_size is None) for token in sentence[:args.max_training_length]],
                stack_features,
                buffer_features,
                passage_seperator,
                min(args.max_training_length, len(sentence)),
                max_stack_size,
                max_buffer_size
            )
    else:
        feature_matrix = np.zeros((num_examples, max_stack_size + max_buffer_size), dtype=np.object if use_elmo else np.int32)

        for index, (stack_features, buffer_features) in enumerate(stack_and_buffer_features):
            add_stack_and_buffer_features(feature_matrix, index, stack_features, buffer_features, max_stack_size, max_buffer_size)
            add_history_features(history_matrix, index, history_features[index], max_hist_size)

    labels = np.array(labels)
    history_lengths = np.array(history_lengths)
    contextualized_embeddings = None
    if use_elmo:
        sentence_lengths = np.array(sentence_lengths)

        # produce contextualized embeddings
        contextualized_embeddings = embedder.sents2elmo(passage_id2sent)
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
