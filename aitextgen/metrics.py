import numpy as np
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
import torch


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
    def __init__(self, name, max_size, n_gram, reference_corpus=None):
        super().__init__(name)
        self._reference_corpus = reference_corpus
        self.max_size = max_size
        self.weights = tuple([1 / n_gram] * n_gram)

    def calculate(self, candidate_sentence):
        score = sentence_bleu(self._reference_corpus, candidate_sentence.tolist())
        return score

    def calculate_batch(self, candidate_sentences):
        sentence_batch = candidate_sentences.tolist()
        score = 0
        for sentence in sentence_batch:
            score += sentence_bleu(self._reference_corpus, sentence, weights=self.weights)
        return score / len(sentence_batch)

    def init_parameters(self, train_token_dataset):
        self._reference_corpus = torch.stack([train_token_dataset.__getitem__(t)
                                              for t in
                                              range(min(train_token_dataset.__len__(), self.max_size))]).tolist()