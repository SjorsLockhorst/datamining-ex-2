from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import MultinomialNB

from extract import create_x_y_unigrams, create_x_y_uni_and_bi


def validate(
    estimator,
    include_bigrams=False,
    scoring=["accuracy", "precision_macro", "recall_macro", "f1_macro"],
):
    if include_bigrams:
        X, y, vectorizer = create_x_y_uni_and_bi()
    else:
        X, y, vectorizer = create_x_y_unigrams()

    result = cross_validate(
        estimator,
        X,
        y,
        scoring=scoring,
    )
    return result


if __name__ == "__main__":
    result_nb = validate(
        MultinomialNB(),
    )
    result_nb_bigram = validate(MultinomialNB(), include_bigrams=True)

    print(result_nb)
    print(result_nb_bigram)
