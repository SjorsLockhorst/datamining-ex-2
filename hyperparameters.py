tree_parameters = {
    n_estimators : [10, 50, 100, 200, 400, 600, 800, 1000],
    ccp_alpha : [0, 0.03, 0.05, 0.07, 0.1, 0.12, 0.15, 0.2, 0.25]
}

rand_forest_parameters = {
    n_estimators : [10, 50, 100, 200, 400, 600, 800, 1000],
    ccp_alpha : [0, 0.03, 0.05, 0.07, 0.1, 0.12, 0.15, 0.2, 0.25]
    max_features : [10, 20, 50, 100]
}

log_regressions_parameters = {
    [1, 2, 3, 4, 5, 10, 20, 30, 40]
}