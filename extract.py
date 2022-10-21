import os
import numpy as np

from config import DECEPTIVE_PATH, TRUTHFUL_PATH


def load_raw_data(dir_path, label):
    all_files = os.listdir(dir_path)
    assert len(all_files) == len(set(all_files))
    all_reviews_raw = []
    for filename in all_files:
        with open(os.path.join(dir_path, filename), "r") as file:
            raw_str = file.read()
            all_reviews_raw.append(raw_str)

    return all_reviews_raw, [label] * len(all_reviews_raw)


if __name__ == "__main__":
    raw_deceptive = load_raw_data(DECEPTIVE_PATH, "deceptive")
    raw_truthful = load_raw_data(TRUTHFUL_PATH, "truthful")

    assert len(raw_deceptive[0]) == 400
    assert len(raw_truthful[0]) == 400
