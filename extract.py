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
