from sklearn.feature_selection import (
    chi2,
    GenericUnivariateSelect,
    VarianceThreshold,
    f_classif,
)
base_param = {            
    "feature_selector__mode": ["percentile"],
    "feature_selector__score_func": [f_classif, chi2]}
tree_parameters = {
    "feature_selector__n_estimators" : [10, 50, 100, 200, 400, 600, 800, 1000],
    "feature_selector__ccp_alpha" : [0, 0.03, 0.05, 0.07, 0.1, 0.12, 0.15, 0.2, 0.25],
    **base_param
}

rand_forest_parameters = {
    "feature_selector__n_estimators" : [10, 50, 100, 200, 400, 600, 800, 1000],
    "feature_selector__ccp_alpha" : [0, 0.03, 0.05, 0.07, 0.1, 0.12, 0.15, 0.2, 0.25],
    "feature_selector__max_features" : [10, 20, 50, 100],
    **base_param
}  

multi_bayes_parameters = {
    "feature_selector__param" : [1, 2, 3, 4, 5, 10, 20, 30, 40],
    **base_param
}

