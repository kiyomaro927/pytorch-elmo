#!/usr/bin/env python
from __future__ import unicode_literals
import codecs

from typing import List

import numpy as np


def pad(sequences: List[List[str]], pad_token: str = '<pad>', pad_left: bool = False):
    """Pad each text sequence to the length of the longest."""
    max_len = max(len(seq) for seq in sequences)
    if pad_left:
        return [[pad_token] * (max_len - len(seq)) + seq for seq in sequences]
    return [seq + [pad_token] * (max_len - len(seq)) for seq in sequences]


def load_embedding_npz(path: str):
    data = np.load(path)
    return [str(w) for w in data['words']], data['vals']


def load_embedding_txt(path: str):
    words, vals = [], []
    with codecs.open(path, 'r', encoding='utf-8') as fin:
        fin.readline()
        for line in fin:
            line = line.strip()
            if line:
                parts = line.split()
                words.append(parts[0])
                vals.append([float(x) for x in parts[1:]])
    return words, np.array(vals)


def load_embedding(path: str):
    if path.endswith(".npz"):
        return load_embedding_npz(path)
    else:
        return load_embedding_txt(path)
