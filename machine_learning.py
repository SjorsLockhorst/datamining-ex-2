from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.feature_selection import (
    chi2,
    GenericUnivariateSelect,
    VarianceThreshold,
    f_classif,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np

from extract import load_raw_data


def validate(X, y, estimator, scoring):

    result = cross_validate(
        estimator,
        X,
        y,
        scoring=scoring,
    )
    return result


def test_uni_and_bi(
    x_uni,
    x_uni_and_bi,
    y,
    estimator_uni,
    estimator_uni_and_bi,
    scoring=["accuracy", "precision_macro", "recall_macro", "f1_macro"],
):
    results_uni = validate(x_uni, y, estimator_uni, scoring)
    results_uni_and_bi = validate(x_uni_and_bi, y, estimator_uni_and_bi, scoring)

    print("Unigram results\n")
    print(results_uni)
    print("\nBigram results \n")
    print(results_uni_and_bi)
    return results_uni, results_uni_and_bi


class CustomVectorizer(CountVectorizer):
    def capital_letter_counts(self, raw_documents):
        return [
            pd.Series(text.split()).str.match(r"[A-Z]").sum() for text in raw_documents
        ]

    def fit_transform(self, raw_documents, y=None, include_captial_counts=True):
        original = super().fit_transform(raw_documents, y=y)
        if include_captial_counts:
            capital_letters = np.array(
                self.capital_letter_counts(raw_documents)
            ).reshape(-1, 1)
            result = np.concatenate((original, capital_letters))

        else:
            result = original
        return result


uni_gram_vectorizer = CustomVectorizer()
uni_and_bigram_vectorizer = CustomVectorizer(ngram_range=(1, 2))

if __name__ == "__main__":

    x_train_raw, y_train, x_test_raw, y_test = load_raw_data()
    x_uni = uni_gram_vectorizer.fit_transform(x_train_raw)
    x_uni_and_bi = uni_and_bigram_vectorizer.fit_transform(x_train_raw)
    x_test_uni = uni_gram_vectorizer.transform(x_test_raw)
    x_test_uni_and_bi = uni_and_bigram_vectorizer.transform(x_test_raw)

    # nb_pipe = Pipeline(
    #     steps=[
    #         ("variance_threshold", VarianceThreshold(0)),
    #         ("feature_selector", GenericUnivariateSelect()),
    #         ("NB", MultinomialNB()),
    #     ]
    # )
    # grid_search_nb = GridSearchCV(
    #     nb_pipe,
    #     {
    #         "feature_selector__mode": ["percentile"],
    #         "feature_selector__param": [1, 2, 3, 4, 5, 10, 20, 30, 40],
    #         "feature_selector__score_func": [f_classif, chi2],
    #     },
    #     scoring="accuracy",
    #     cv=10,
    #     n_jobs=-1,
    # )
    # grid_search_nb.fit(x_uni, y_train)
    # print(grid_search_nb.best_params_)
    # print(grid_search_nb.best_score_)

    # grid_search_nb.fit(x_uni_and_bi, y_train)
    # print(grid_search_nb.best_params_)
    # print(grid_search_nb.best_score_)
    # clf = LogisticRegressionCV(
    #     cv=10, penalty="l1", solver="liblinear", n_jobs=-1, scoring="accuracy"
    # )
    # clf_uni = clf.fit(x_uni, y_train)
    # print(clf_uni.score(x_test_uni, y_test))
    # clf_uni_and_bi = clf.fit(x_uni_and_bi, y_train)
    # print(clf_uni_and_bi.score(x_test_uni_and_bi, y_test))

    # X_new = SelectPercentile(chi2, percentile=10).fit_transform(x_uni, y_train)
    # print(X_new.shape)
    # test_uni_and_bi(x_uni, x_uni_and_bi, y_train, MultinomialNB(), MultinomialNB())
    # test_uni_and_bi(
    #     x_uni, x_uni_and_bi, y_train, LogisticRegression(), LogisticRegression()
    # )
    # test_uni_and_bi(
    #     x_uni, x_uni_and_bi, y_train, DecisionTreeClassifier(), DecisionTreeClassifier()
    # )
