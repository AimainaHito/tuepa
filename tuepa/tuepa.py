import sys
import time
from collections import defaultdict

import concurrent.futures
import os
from enum import Enum
from functools import partial
from glob import glob
from semstr.convert import FROM_FORMAT, TO_FORMAT, from_text
from semstr.evaluate import EVALUATORS, Scores
from semstr.util.amr import LABEL_ATTRIB, WIKIFIER
from tqdm import tqdm
from ucca import diffutil, ioutil, textutil, layer0, layer1
from ucca.evaluation import LABELED, UNLABELED, EVAL_TYPES, evaluate as evaluate_ucca
from ucca.normalization import normalize


from numberer import Numberer
from oracle import Oracle
from states.state import State

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

w_numberer = Numberer()
nt_numberer = Numberer()
t_numberer = Numberer()


def extract_features(state, w_numberer, n_t_numberer, train=True):
    stack = state.stack
    buffer = state.buffer
    stack_features = [w_numberer.number(e,train=train) if e.paragraph else n_t_numberer.number(e,train=train) for e in stack]

    buffer_features = [w_numberer.number(e,train=train) if e.paragraph else n_t_numberer.number(e,train=train) for e in buffer]
    return stack_features, buffer_features


features = []
labels = []

from sklearn.linear_model import LogisticRegression

max_s = max_b = -1

for passage in read_passages(["../test_files/*xml"]):
    s = State(passage)
    o = Oracle(passage)
    while not s.finished:
        actions = o.generate_actions(state=s)
        a = next(actions)
        stack_features, buffer_features = extract_features(s, w_numberer, nt_numberer)
        label = a.type_id
        s.transition(a)
        a.apply()
        features.append((stack_features,buffer_features))
        max_s = max(len(stack_features), max_s)
        max_b = max(len(buffer_features), max_b)
        labels.append(label)

for n,feature in enumerate(features):
    while len(feature[0]) != max_s:
        feature[0].append(0)
    while len(feature[1]) != max_b:
        feature[1].append(0)
    features[n] = feature[0]+feature[1]

lr = LogisticRegression()
lr.fit(features,labels)

