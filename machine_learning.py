from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from extract import create_x_y_unigrams, create_x_y_uni_and_bi


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


if __name__ == "__main__":
    x_unigrams, y, _ = create_x_y_unigrams()
    x_uni_and_bi, y, _ = create_x_y_uni_and_bi()

    test_uni_and_bi(x_unigrams, x_uni_and_bi, y, MultinomialNB(), MultinomialNB())
    test_uni_and_bi(
        x_unigrams, x_uni_and_bi, y, LogisticRegression(), LogisticRegression()
    )
    test_uni_and_bi(
        x_unigrams, x_uni_and_bi, y, DecisionTreeClassifier(), DecisionTreeClassifier()
    )
