#!/usr/bin/env python
from __future__ import print_function
from __future__ import unicode_literals
import os
import codecs
import random
import logging
import json

from typing import List, Dict, Optional

import torch
from .modules.embedding_layer import EmbeddingLayer
from .utils import dict2namedtuple
from .frontend import create_one_batch
from .frontend import Model

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s: %(message)s')


def read_list(sents: List[List[str]], max_chars: Optional[int] =None):
    """Read a list of word lists.

    Note: max_chars, the number of maximum characters in a word, is used
    when the model is configured with CNN word encoder.

    """
    dataset = []
    for sent in sents:
        data = ['<bos>']
        for token in sent:
            if max_chars is not None and len(token) + 2 > max_chars:
                token = token[:max_chars - 2]
            data.append(token)
        data.append('<eos>')
        dataset.append(data)
    return dataset, sents


def recover(li, ind):
    # li[piv], ind = torch.sort(li[piv], dim=0, descending=(not unsort))
    dummy = list(range(len(ind)))
    dummy.sort(key=lambda l: ind[l])
    li = [li[i] for i in dummy]
    return li


# shuffle training examples and create mini-batches
def create_batches(x: List[List[str]],
                   batch_size: int,
                   word2id: Dict[str, int],
                   char2id: Dict[str, int],
                   config,
                   perm: Optional[List[int]] = None,
                   shuffle: bool = False,
                   sort: bool = True,
                   text: Optional[List[List[str]]] = None):
    ind = list(range(len(x)))
    lst = perm or list(range(len(x)))
    if shuffle:
        random.shuffle(lst)

    if sort:
        lst.sort(key=lambda l: -len(x[l]))

    x = [x[i] for i in lst]
    ind = [ind[i] for i in lst]
    if text is not None:
        text = [text[i] for i in lst]

    sum_len = 0.0
    batches_w, batches_c, batches_lens, batches_masks, batches_text, batches_ind = [], [], [], [], [], []
    size = batch_size
    nbatch = (len(x) - 1) // size + 1
    for i in range(nbatch):
        start_id, end_id = i * size, (i + 1) * size
        bw, bc, blens, bmasks = create_one_batch(x[start_id: end_id], word2id, char2id, config, sort=sort)
        sum_len += sum(blens)
        batches_w.append(bw)
        batches_c.append(bc)
        batches_lens.append(blens)
        batches_masks.append(bmasks)
        batches_ind.append(ind[start_id: end_id])
        if text is not None:
            batches_text.append(text[start_id: end_id])

    if sort:
        perm = list(range(nbatch))
        random.shuffle(perm)
        batches_w = [batches_w[i] for i in perm]
        batches_c = [batches_c[i] for i in perm]
        batches_lens = [batches_lens[i] for i in perm]
        batches_masks = [batches_masks[i] for i in perm]
        batches_ind = [batches_ind[i] for i in perm]
        if text is not None:
            batches_text = [batches_text[i] for i in perm]

    recover_ind = [item for sublist in batches_ind for item in sublist]
    if text is not None:
        return batches_w, batches_c, batches_lens, batches_masks, batches_text, recover_ind
    return batches_w, batches_c, batches_lens, batches_masks, recover_ind


class Embedder:

    def __init__(self, model_dir: str, batch_size: int = 64) -> None:
        self.model_dir = model_dir
        self.batch_size = batch_size

        self.use_cuda = False
        self.char_lexicon = None
        self.word_lexicon = None

        self.model, self.config = self.get_model()

    def __call__(self, *args, **kwargs):
        return self.sents2elmo(*args, **kwargs)

    def get_model(self):
        logging.info(f'Building ELMo...')
        self.use_cuda = torch.cuda.is_available()
        # load the model configurations
        args2 = dict2namedtuple(json.load(codecs.open(
            os.path.join(self.model_dir, 'config.json'), 'r', encoding='utf-8')))

        with open(os.path.join(self.model_dir, args2.config_path), 'r') as fin:
            config = json.load(fin)

        # For the model trained with character-based word encoder.
        char_emb_layer = None
        if config['token_embedder']['char_dim'] > 0:
            self.char_lexicon = {}
            with codecs.open(os.path.join(self.model_dir, 'char.dic'), 'r', encoding='utf-8') as fpi:
                for line in fpi:
                    tokens = line.strip().split('\t')
                    if len(tokens) == 1:
                        tokens.insert(0, '\u3000')
                    token, i = tokens
                    self.char_lexicon[token] = int(i)
            char_emb_layer = EmbeddingLayer(
                config['token_embedder']['char_dim'], self.char_lexicon, fix_emb=False, embs=None)
            logging.info(f'char embedding size: {len(char_emb_layer.word2id)}')

        # For the model trained with word form word encoder.
        word_emb_layer = None
        if config['token_embedder']['word_dim'] > 0:
            self.word_lexicon = {}
            with codecs.open(os.path.join(self.model_dir, 'word.dic'), 'r', encoding='utf-8') as fpi:
                for line in fpi:
                    tokens = line.strip().split('\t')
                    if len(tokens) == 1:
                        tokens.insert(0, '\u3000')
                    token, i = tokens
                    self.word_lexicon[token] = int(i)
            word_emb_layer = EmbeddingLayer(
                config['token_embedder']['word_dim'], self.word_lexicon, fix_emb=False, embs=None)
            logging.info(f'word embedding size: {len(word_emb_layer.word2id)}')

        model = Model(config, word_emb_layer, char_emb_layer, self.use_cuda)
        if self.use_cuda:
            model.cuda()
        logging.info(str(model))
        model.load_model(self.model_dir)
        model.eval()  # configure the model to evaluation mode.
        return model, config

    def sents2elmo(self, sents: List[List[str]], output_layer: int = -1):
        read_function = read_list
        if self.config['token_embedder']['name'].lower() == 'cnn':
            test, text = read_function(sents, self.config['token_embedder']['max_characters_per_token'])
        else:
            test, text = read_function(sents)

        # create test batches from the input data.
        test_w, test_c, test_lens, test_masks, test_text, recover_ind = create_batches(
            test, self.batch_size, self.word_lexicon, self.char_lexicon, self.config, text=text)

        after_elmo = []
        for w, c, lens, masks, texts in zip(test_w, test_c, test_lens, test_masks, test_text):
            output = self.model.forward(w, c, masks)
            for i, text in enumerate(texts):
                if self.config['encoder']['name'].lower() == 'lstm':
                    data = output[i, 1:lens[i]-1, :].data
                elif self.config['encoder']['name'].lower() == 'elmo':
                    data = output[:, i, 1:lens[i]-1, :].data

                if self.use_cuda:
                    data = data.cpu()
                data = data.numpy()

                payload = data if output_layer == -1 else data[output_layer]
                after_elmo.append(payload)
        return recover(after_elmo, recover_ind)
