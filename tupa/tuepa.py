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

from tupa.__version__ import GIT_VERSION
from tupa.config import Config, Iterations
from tupa.model import Model, NODE_LABEL_KEY, ClassifierProperty
from tupa.oracle import Oracle
from tupa.states.state import State
from tupa.traceutil import set_traceback_listener

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


for passage in read_passages(["/data/research/tuepa/test_files/504.xml"]):
    print(passage)
    print("\nState:")
    s = State(passage)
    print(s)
    print("\nOracle:")
    o = Oracle(passage)
    while not s.finished:
        actions = o.generate_actions(state=s)
        a = next(actions)
        print("action:" ,a)
        s.transition(a)
        a.apply()
        print(s)
        print()
    # import IPython; IPython.embed()
