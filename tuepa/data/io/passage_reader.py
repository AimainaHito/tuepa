from collections import defaultdict
import xml.etree.ElementTree as ET
from ucca.convert import file2passage, passage2file, from_text, to_text, split2segments, from_standard

DEFAULT_LANG = "en"
DEFAULT_ATTEMPTS = 3
DEFAULT_DELAY = 5


class SingleFileLazyLoadedPassages:
    """
    Iterable interface to Passage objects that loads files on-the-go and can be iterated more than once
    """
    def __init__(self, file, sentences=False, paragraphs=False, converters=None, lang=DEFAULT_LANG,
                 attempts=DEFAULT_ATTEMPTS, delay=DEFAULT_DELAY):
        self.files = file
        self.sentences = sentences
        self.paragraphs = paragraphs
        self.split = self.sentences or self.paragraphs
        self.converters = defaultdict(lambda: from_text) if converters is None else converters
        self.lang = lang
        self.attempts = attempts
        self.delay = delay
        self._files_iter = None
        self._split_iter = None
        self._file_handle = None

    def __iter__(self):
        with open(self.files, "r") as f:
            yield from map(from_standard,ET.ElementTree().parse(f))

    # The following three methods are implemented to support shuffle;
    # note files are shuffled but there is no shuffling within files, as it would not be efficient.
    # Note also the inconsistency because these access the files while __iter__ accesses individual passages.
    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        return self.files[i]

    def __setitem__(self, i, value):
        self.files[i] = value

    def __bool__(self):
        return bool(self.files)