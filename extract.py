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

    raw_deceptive_train, deceptive_labels_train = _load_raw_data(
        os.path.join(DECEPTIVE_PATH, "train"), "deceptive"
    )
    raw_deceptive_test, deceptive_labels_test = _load_raw_data(
        os.path.join(DECEPTIVE_PATH, "test"), "deceptive"
    )
    raw_truthful_train, truthful_labels_train = _load_raw_data(
        os.path.join(TRUTHFUL_PATH, "train"), "truthful"
    )
    raw_truthful_test, truthful_labels_test = _load_raw_data(
        os.path.join(TRUTHFUL_PATH, "test"), "truthful"
    )
    x_raw_train = raw_deceptive_train + raw_truthful_train
    y_train = deceptive_labels_train + truthful_labels_train

    x_raw_test = raw_deceptive_test + raw_truthful_test
    y_test = deceptive_labels_test + truthful_labels_test
    assert len(x_raw_train) == 640
    assert len(y_train) == 640

    assert len(x_raw_test) == 160
    assert len(y_test) == 160

    return x_raw_train, y_train, x_raw_test, y_test


def create_x_unigrams(x_raw):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(x_raw).toarray()
    return X, vectorizer


def create_x_bigrams(x_raw):
    vectorizer = CountVectorizer(ngram_range=(2, 2))
    X = vectorizer.fit_transform(x_raw).toarray()
    return X, vectorizer


def create_x_y_unigrams():
    x_train, y_train, _, _ = load_raw_data()
    X, vect = create_x_unigrams(x_train)
    return X, y_train, vect


def create_x_y_uni_and_bi():
    x_train, y_train, _, _ = load_raw_data()
    vectorizer = CountVectorizer(ngram_range=(1, 2))
    X = vectorizer.fit_transform(x_train).toarray()
    return X, y_train, vectorizer
