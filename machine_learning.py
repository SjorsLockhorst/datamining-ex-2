from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.feature_selection import (
    chi2,
    GenericUnivariateSelect,
    VarianceThreshold,
    f_classif,
)
from sklearn.metrics import precision_recall_fscore_support
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import hyperparameters as param
from extract import load_raw_data
from stop_words import stop_words
import evaluate


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


class CustomVectorizer(BaseEstimator, TransformerMixin):
    def capital_letter_counts(self, raw_documents):
        return [
            pd.Series(text.split()).str.match(r"[A-Z]").sum() for text in raw_documents
        ]

    def transform(self, raw_documents, y=None):
        """The workhorse of this feature extractor"""
        return self.capital_letter_counts(raw_documents)

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self


uni_gram_vectorizer = CountVectorizer(stop_words="english")
uni_and_bigram_vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words=stop_words)
vect = CustomVectorizer()
uni_all_feats = FeatureUnion(
    [("bag-of-words", uni_gram_vectorizer), ("capital-letters", uni_gram_vectorizer)]
)
uni_and_bi_all_feats = FeatureUnion(
    [
        ("bag-of-words", uni_and_bigram_vectorizer),
        ("capital-letters", uni_gram_vectorizer),
    ]
)

if __name__ == "__main__":
    x_train_raw, y_train, x_test_raw, y_test = load_raw_data()
    x_uni = uni_all_feats.fit_transform(x_train_raw)
    x_uni_and_bi = uni_and_bi_all_feats.fit_transform(x_train_raw)
    x_test_uni = uni_all_feats.transform(x_test_raw)
    x_test_uni_and_bi = uni_and_bi_all_feats.transform(x_test_raw)
    print(x_uni.shape)  
    print(x_uni_and_bi.shape)
    nb_pipe = Pipeline(
        steps=[
            ("variance_threshold", VarianceThreshold(0)),
            ("feature_selector", GenericUnivariateSelect()),
            #("NB", MultinomialNB()),
            #("rand_forest", RandomForestClassifier(random_state=2))
            ("tree", RandomForestClassifier(max_features = None, random_state=2))
            #("log_regression",  LogisticRegression(random_state = 2))
        ],
        #verbose = 1
    )
    grid_search_nb = GridSearchCV(
        nb_pipe,
        param.tree_parameters,
        scoring="accuracy",
        cv=10,
        n_jobs=-1,
    )
    """
    grid_search_nb.fit(x_uni, y_train)
    print(grid_search_nb.best_params_)
    print(grid_search_nb.best_score_)

    grid_search_nb.fit(x_uni_and_bi, y_train)
    print(grid_search_nb.best_params_)
    print(grid_search_nb.best_score_)
    
    var_thr = VarianceThreshold(threshold = 0) #Removing both constant and quasi-constant
    var_thr.fit(x_uni)
    bools = var_thr.get_support()
    print(x_uni.shape, bools.shape)
    x_uni = x_uni[:,bools]
    """
    #models with best parameters uni:
    mn_uni = MultinomialNB() #param: 11.28837891684689, score func f_classif (0.8375)
    tree_uni = RandomForestClassifier(1,random_state = 2, ccp_alpha=0.025) #0.7125
    rand_uni = RandomForestClassifier(2000,random_state = 2, ccp_alpha=0, max_features= 88) #(0.8453125)
    log_uni = LogisticRegression(random_state = 2, C = 265608.7782946684)
    
    mn_bi = MultinomialNB()
    tree_bi = RandomForestClassifier(1,random_state = 2, ccp_alpha=0.025)
    rand_bi = RandomForestClassifier(2000,random_state = 2, ccp_alpha=0.005, max_features=50)
    log_bi = LogisticRegression(random_state = 2, C = 5455.594781168515)
    models = [tree_uni, rand_uni, log_uni, mn_uni, tree_bi, rand_bi, log_bi, mn_bi]

    univariate_uni = GenericUnivariateSelect(f_classif, mode='percentile', param=11.28837891684689)
    nbdata_uni = univariate_uni.fit_transform(x_uni, y_train)
    nbtest_uni = univariate_uni.transform(x_test_uni)
    univariate_bi = GenericUnivariateSelect(f_classif, mode='percentile', param=5.455594781168519)
    nbdata_bi = univariate_bi.fit_transform(x_uni_and_bi, y_train)
    nbtest_bi = univariate_bi.transform(x_test_uni_and_bi)
    predictions = np.empty((len(models),len(y_test)), dtype= str)
    for i in range(len(models)):
        if(i == 3):
            models[i].fit(nbdata_uni, y_train)
            ypred = models[i].predict(nbtest_uni)
        elif(i==7):
            models[i].fit(nbdata_bi, y_train)
            ypred = models[i].predict(nbtest_bi)
        elif(i<4):
            models[i].fit(x_uni, y_train)
            ypred = models[i].predict(x_test_uni)
        else:
            models[i].fit(x_uni_and_bi, y_train)
            ypred = models[i].predict(x_test_uni_and_bi)
        predictions[i] = ypred
        print(evaluate.model_accuracy(ypred, y_test))
        print(precision_recall_fscore_support(y_test, ypred , average = 'binary', pos_label='truthful'))

    print("mcNemar scores")
    modelnames = ["unitree", "uniforest", "unilog", "unibayes", "bitree", "biforest", "bilog", "bibayes"]
    for i in range(len(predictions)):
        for j in range(len(predictions)):
            if(i!=j):
                print("{} vs {}".format(modelnames[i], modelnames[j]))
                y_test = [i=='truthful' for i in y_test]
                evaluate.mcnemar_test(predictions[i]=='t', predictions[j]=='t', y_test)
        
    #LogisticRegression(random_state = 2, C = 265608.7782946684) (0.8375)
    #models with best parameters bi:
    #MultinomialNB() param:5.455594781168519, score func f_classif (0.85)
    #clf = RandomForestClassifier(1,random_state = 2, ccp_alpha=0.025)
    #clf = RandomForestClassifier(2000,random_state = 2, ccp_alpha=0.005, max_features=50) (0.85625) 2500 50 0.859375
    #LogisticRegression(random_state = 2, C = 5455.594781168515) (0.8421875)
    clf = LogisticRegressionCV(
         cv=10, penalty="l1", solver="liblinear", n_jobs=-1, scoring="accuracy"
    )
    #clf_uni = clf.fit(x_uni, y_train)
    #print(clf_uni.score(x_uni, y_train))
    #print(clf_uni.score(x_test_uni, y_test))
    #clf_uni_and_bi = clf.fit(x_uni_and_bi, y_train)
    #print(clf_uni_and_bi.score(x_test_uni_and_bi, y_test))

    # X_new = SelectPercentile(chi2, percentile=10).fit_transform(x_uni, y_train)
    # print(X_new.shape)
    # test_uni_and_bi(x_uni, x_uni_and_bi, y_train, MultinomialNB(), MultinomialNB())
    # test_uni_and_bi(
    #     x_uni, x_uni_and_bi, y_train, LogisticRegression(), LogisticRegression()
    # )
    # test_uni_and_bi(
    #     x_uni, x_uni_and_bi, y_train, DecisionTreeClassifier(), DecisionTreeClassifier()
    # )
