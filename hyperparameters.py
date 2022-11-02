from sklearn.feature_selection import (
    chi2,
    GenericUnivariateSelect,
    VarianceThreshold,
    f_classif,
)
import numpy as np
base_param = {            
    "feature_selector__mode": ["percentile"],
    "feature_selector__score_func": [f_classif, chi2],
    "feature_selector__param" : np.logspace(0,2, 20)#[1, 2, 3, 4, 5, 10, 20, 30, 40]
    }

tree_parameters = {
    "tree__n_estimators" : [1],#, 100, 200, 400, 600, 800, 1000],
    "tree__ccp_alpha" : [0,0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]#, 0.07, 0.1, 0.12, 0.15, 0.2, 0.25],
    #**base_param
}

rand_forest_parameters = {
    "rand_forest__n_estimators" : [200, 400, 500, 600,700, 800, 900, 1000],#[10, 50, 100, 200, 400, 500, 600,700, 800, 900, 1000],
    "rand_forest__ccp_alpha" : [0, 0.025],#, 0.05, 0.07, 0.1, 0.12, 0.15, 0.2, 0.25],
    "rand_forest__max_features" : [10, 20, 50, 100, 'sqrt', 400],
    #**base_param
}  
log_regress_parameters ={
    "log_regression__C": [265608.7782946684] #np.logspace(-5, 7, 100),
}
multi_bayes_parameters = base_param

