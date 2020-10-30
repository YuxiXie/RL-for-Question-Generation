import torch
import functools
from tqdm import tqdm

import onqg.dataset.Constants as Constants

from transformers import AutoTokenizer


class Vocab(object):
    """
    Class for vocabulary
    contain BERTTokenizer 
    Default is defined by given functions
    """
    def __init__(self, special_words, lower=True, opts=None):
        self.type = opts if opts else 'default'
        self.lower = lower
        self.special_words = special_words        

        if self.type == 'default':
            self.special = []
            self.idxToLabel, self.labelToIdx, self.frequencies = {}, {}, {}
            if len(special_words) > 0:
                self.addSpecials([v for _,v in special_words.items()])
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.type)
            self.tokenizer.add_special_tokens(special_words)
            self.labelToIdx = functools.partial(self.tokenizer.convert_tokens_to_ids)
            self.idxToLabel = functools.partial(self.tokenizer.convert_ids_to_tokens)            
            self.special = self.labelToIdx([v for _,v in special_words.items()])
            self.frequencies = {k:100 for k in self.special}

    @classmethod
    def from_opt(cls, corpus=None, opt=None, pretrained=''):
        special_words = {'pad_token': Constants.PAD_WORD, 'unk_token': Constants.UNK_WORD}
        if opt['transf']:
            special_words['cls_token'] = Constants.CLS_WORD
        if opt['tgt'] and not pretrained:
            special_words['bos_token'], special_words['eos_token'] = Constants.BOS_WORD, Constants.EOS_WORD
        if opt['separate']:
            special_words['sep_token'] = Constants.SEP_WORD
        
        if pretrained:
            vocab = cls(special_words, opts=pretrained)
        else:
            assert corpus is not None and opt is not None
            vocab = cls(special_words, lower=opt['lower'])
            assert special_words == vocab.special_words
            for sent in corpus:
                for word in sent:
                    vocab.add(word, lower=opt['lower'])
            original_size = vocab.size
            vocab = vocab.prune(opt['size'], opt['frequency'], opt['mode'])
            print("Truncate vocabulary size from " + str(original_size) + " to " + str(vocab.size))
        
        return vocab

    @property
    def size(self):
        if self.type.count('bert'):
            return len(self.tokenizer.vocab)
        return len(self.idxToLabel)

    def lookup(self, key, default=Constants.UNK):
        try:
            return self.labelToIdx([key])[0] if self.type != 'default' else self.labelToIdx[key]
        except KeyError:
            return default

    def getLabel(self, idx, default=Constants.UNK_WORD):
        try:
            return self.idxToLabel([idx])[0] if self.type != 'default' else self.idxToLabel[idx]
        except KeyError:
            return default

    # Mark this `label` and `idx` as special (i.e. will not be pruned).
    def addSpecial(self, label, idx=None):
        idx = self.add(label, idx, lower=False)
        self.special += [idx]

    # Mark all labels in `labels` as specials (i.e. will not be pruned).
    def addSpecials(self, labels):
        for label in labels:
            self.addSpecial(label, idx=None)

    # Add `label` in the dictionary. Use `idx` as its index if given.
    def add(self, label, idx=None, lower=False):
        assert self.type != 'bert', "BERT has already been pretrained"

        lower = self.lower if lower and label not in self.special_words else False
        label = label.lower() if lower else label
        if idx is not None:
            self.idxToLabel[idx] = label
            self.labelToIdx[label] = idx
        else:
            if label in self.labelToIdx:
                idx = self.labelToIdx[label]
            else:
                idx = len(self.idxToLabel)
                self.idxToLabel[idx] = label
                self.labelToIdx[label] = idx

        if idx not in self.frequencies:
            self.frequencies[idx] = 1
        else:
            self.frequencies[idx] += 1

        return idx

    # Return a new dictionary with the `size` most frequent entries.
    def prune(self, size, frequency, mode='size'):
        assert self.type == 'default', self.type.upper() + " has already been pretrained"

        freq = torch.Tensor([self.frequencies[i] for i in range(len(self.frequencies))])
        freq, idx = torch.sort(freq, 0, True)

        newVocab = Vocab([self.idxToLabel[i] for i in self.special], lower=self.lower)
        
        if mode == 'size':
            if size >= self.size:
                return self
            # Only keep the `size` most frequent entries.
            for i in idx[:size]:
                newVocab.add(self.idxToLabel[i.item()])
            return newVocab
        elif mode == 'frequency':
            if frequency <= 1:
                return self
            for cnt, i in enumerate(idx):
                if freq[cnt] < frequency:
                    return newVocab
                newVocab.add(self.idxToLabel[i.item()])
                newVocab.frequencies[i.item()] = self.frequencies[i.item()]
        else:
            print("mode error in Vocab.prune! ")
            assert False
        
        return newVocab

    # Convert `labels` to indices. Use `unkWord` if not found.
    # Optionally insert `bosWord` at the beginning and `eosWord` at the .
    def convertToIdx(self, labels, unkWord=Constants.UNK_WORD):
       unk = self.lookup(unkWord)
       indexes = [self.lookup(label, default=unk) for label in labels]
       return torch.LongTensor(indexes)

    # Convert `idx` to labels. If index `stop` is reached, convert it and return.
    def convertToLabels(self, idx, stopList=[Constants.PAD, Constants.EOS]):
        labels = [self.getLabel(i) for i in idx if i not in stopList]
        return labels

    def word_count(self, corpus):
        for sent in tqdm(corpus, desc='   - (counting words) -   '):
            for word in sent:
                word = word.item() if isinstance(word, torch.Tensor) else word
                if word not in self.frequencies:
                    self.frequencies[word] = 0
                self.frequencies[word] += 1