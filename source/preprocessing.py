from typing import Dict

import numpy as np

from primitives import BooleanMatrix


class Tokenizer:
    def __init__(self):
        self.encoder = dict()
        self.next_code = 0
        self.decoder = []

    def encode(self, term):
        code = self.encoder.setdefault(term, self.next_code)

        if code == self.next_code:
            self.next_code += 1
            self.decoder.append(term)

        return code

    def decode(self, code):
        if code < 0 or code >= len(self.decoder):
            return None

        return self.decoder[code]

    def __len__(self):
        return self.next_code


class Terms:
    def __init__(self, tokenizer: Tokenizer, term_occurrence: BooleanMatrix):
        self.dataset = term_occurrence
        self.tokenizer = tokenizer

    def term_frequency(self):
        return np.mean(self.dataset, axis=1)

    def term_labels(self):
        return self.tokenizer.decoder[:]

    def term_occurrence(self):
        return self.dataset


class DataDecoder:
    @classmethod
    def from_term_sets(cls, data: Dict) -> Terms:
        term_coding = cls._build_term_coding(data)
        term_occurrence = cls._build_term_occurrence(data, term_coding)
        return Terms(term_coding, term_occurrence)

    @classmethod
    def _build_term_coding(cls, data):
        term_coding = Tokenizer()

        for sample in data:
            for term in sample["topics"]:
                term_coding.encode(term)

        return term_coding

    @classmethod
    def _build_term_occurrence(cls, data, term_coding):
        num_sample = len(data)
        num_terms = len(term_coding)

        term_occurrence = np.zeros((num_terms, num_sample), dtype=np.bool)

        for i, sample in enumerate(data):
            for term in sample["topics"]:
                term_occurrence[term_coding.encode(term), i] = True

        return term_occurrence
