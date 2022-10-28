import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    ConfusionMatrixDisplay,
    confusion_matrix,
)
from scipy.stats.contingency import chi2_contingency
from scipy.stats import binomtest, chi2
def chi_2(table, verbose=True):
    chi_2, p, dof, expected = chi2_contingency(table)
    if verbose:
        print(f"Chi2: {chi_2}")
        print(f"p: {p}")
        print(f"df: {dof}")
        print(f"expected: \n{expected}")
    return chi_2, p, dof, expected

def model_accuracy(test_y, y_pred, verbose=False):
    #y_pred = pred_model(test_x, test_y, model, verbose=verbose)
    return accuracy_score(test_y, y_pred)

def paired_bootstrap(test_x, test_y, model_a, model_b, b):
    test_x = test_x.to_numpy()
    test_y = test_y.to_numpy()
    acc_a = model_accuracy(test_x, test_y, model_a)
    acc_b = model_accuracy(test_x, test_y, model_b)
    diff_x = acc_a - acc_b

    a_better_count = 0
    for i in range(b):
        if i % 100 == 0:
            print(f"Completed {i/b}%")
        indices = np.random.choice(len(test_x), len(test_x), replace=True)
        random_x = test_x[indices]
        corresponding_y = test_y[indices]
        acc_a_i = model_accuracy(random_x, corresponding_y, model_a)
        acc_b_i = model_accuracy(random_x, corresponding_y, model_b)
        diff_x_i = acc_a_i - acc_b_i
        if diff_x_i - diff_x >= diff_x:
            a_better_count += 1
            print("HIT")
    p_val = a_better_count / b
    return p_val

def create_comparison_table(pred_y_a, pred_y_b, correct):
    correct_a = pred_y_a == correct
    correct_b = pred_y_b == correct
    return confusion_matrix(correct_a, correct_b)



def mcnemar_test(pred_y_a, pred_y_b, correct):
    table = create_comparison_table(pred_y_a, pred_y_b, correct)
    print(table)
    b = table[0][1]
    c = table[1][0]
    x2 = (b - c) ** 2 / (b + c)
    p = chi2.sf(x2, df=1)
    print(x2)
    print(1)
    print(p)
