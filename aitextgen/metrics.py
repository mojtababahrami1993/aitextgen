import numpy as np
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
import torch
import time


class Metric():
    def __init__(self, name):
        self.name = name

    def init_parameters(self, train_token_dataset):
        pass

    def calculate(self):
        pass

    def calculate_batch(self):
        pass


class BELU(Metric):
    def __init__(self, name, max_size, n_gram, reference_corpus=[]):
        super().__init__(name)
        self._reference_corpus = reference_corpus
        self.max_size = max_size
        self.weights = tuple([1 / n_gram] * n_gram)
        self._eos_token = None

    def calculate(self, candidate_sentence):
        str_candidate_sentence = [str(w) for w in candidate_sentence]
        clean_candidate_sentence = ' '.join(str_candidate_sentence).split(self._eos_token)[0].split()
        score = sentence_bleu(self._reference_corpus, clean_candidate_sentence, weights=self.weights)
        return score

    def calculate_batch(self, candidate_sentences):
        sentence_batch = candidate_sentences.tolist()
        score = 0
        for sentence in sentence_batch:
            score += self.calculate(sentence)
        return score / len(sentence_batch)

    def init_parameters(self, train_token_dataset, eos_token_id):
        train_sentenses = ' '.join(np.char.mod('%ld', train_token_dataset)).split(str(eos_token_id))
        train_sentenses = [s.split(' ') for s in train_sentenses]
        for idx, sent in enumerate(train_sentenses):
            self._reference_corpus.append([w for w in sent if len(w) > 0])
            if idx > self.max_size:
                break
        self._eos_token = str(eos_token_id)

