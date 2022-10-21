import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from config import DECEPTIVE_PATH, TRUTHFUL_PATH


def load_raw_data():
    def _load_raw_data(dir_path, label):
        all_files = os.listdir(dir_path)
        assert len(all_files) == len(set(all_files))
        all_reviews_raw = []
        for filename in all_files:
            with open(os.path.join(dir_path, filename), "r") as file:
                raw_str = file.read()
                all_reviews_raw.append(raw_str)

        return all_reviews_raw, [label] * len(all_reviews_raw)

    raw_deceptive, deceptive_labels = _load_raw_data(DECEPTIVE_PATH, "deceptive")
    raw_truthful, truthful_labels = _load_raw_data(TRUTHFUL_PATH, "truthful")
    x_raw = raw_deceptive + raw_truthful
    y = deceptive_labels + truthful_labels
    return x_raw, y


def create_x_unigrams(x_raw):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(x_raw).toarray()
    return X, vectorizer


def create_x_bigrams(x_raw):
    vectorizer = CountVectorizer(ngram_range=(2, 2))
    X = vectorizer.fit_transform(x_raw).toarray()
    return X, vectorizer


def create_x_y_unigrams():
    x_raw, y = load_raw_data()
    X, vect = create_x_unigrams(x_raw)
    return X, y, vect


def create_x_y_uni_and_bi():
    x_raw, y = load_raw_data()
    vectorizer = CountVectorizer(ngram_range=(1, 2))
    X = vectorizer.fit_transform(x_raw).toarray()
    return X, y, vectorizer
