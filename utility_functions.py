import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
np.random.seed(32)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder,  MaxAbsScaler, RobustScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from lightgbm import LGBMClassifier

import math


# With this fucntion you can quickle see your model's performance
def classification_results(y_true, y_pred_proba, name="", threshold=0.5):
    proba_to_label = lambda proba: 1 if proba>threshold else 0
    y_pred_proba = pd.Series(y_pred_proba)
    
    roc_auc = metrics.roc_auc_score(y_true, y_pred_proba)
    accuracy = metrics.accuracy_score(y_true, y_pred_proba.apply(proba_to_label))
    precision = metrics.precision_score(y_true, y_pred_proba.apply(proba_to_label))
    recall = metrics.recall_score(y_true, y_pred_proba.apply(proba_to_label))
    f1_score = metrics.f1_score(y_true, y_pred_proba.apply(proba_to_label))
    tn_norm, fp_norm, fn_norm, tp_norm = metrics.confusion_matrix(y_true, y_pred_proba.apply(proba_to_label), normalize='all').ravel()
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred_proba.apply(proba_to_label)).ravel()
    
    result = pd.DataFrame({
        'roc_auc': [roc_auc],
        'accuracy': [accuracy],
        'precision': [precision],
        'recall': [recall],
        'f1_score': [f1_score],
        'tn_norm': [tn_norm],
        'fp_norm': [fp_norm],
        'fn_norm': [fn_norm],
        'tp_norm': [tp_norm],
        'tn': [tn],
        'fp': [fp],
        'fn': [fn],
        'tp': [tp]},
        index=[name]).round(3)

    return result

# This is a wrapper function to use classification_results function for test dataset.
# You should not access to test labels outside of this function
def classification_results_test(y_pred_proba, threshold=0.5, name="Test"):
    y_test = pd.read_csv("https://raw.githubusercontent.com/erdemsirel/data_science_case/main/credit_risk_test_label.csv")['label']
    test_result = classification_results(y_true=y_test, y_pred_proba=y_pred_proba, name=name, threshold=threshold)
    tn = test_result.loc[name, 'tn']
    fp = test_result.loc[name, 'fp']
    fn = test_result.loc[name, 'fn']
    tp = test_result.loc[name, 'tp']
    predicted_positive = tp + fp
    excess_predicted_positive = math.ceil((predicted_positive-2000)/75) if  predicted_positive > 2000 else 0
    fail_to_identify_minimum_fine = 500000 if tp < 500 else 0

    test_result["Cost"] = excess_predicted_positive * 13000 + fn * 500 + fail_to_identify_minimum_fine
    return test_result

# This is a wrapper function to evaluate train, validation & test dats all together. 
# Here we also calculate overfit.
def classification_results_combined(y_train, y_train_pred_proba, 
                                    y_val, y_val_pred_proba,
                                    y_test_pred_proba,
                                    threshold=0.5,
                                    print_overfit=True):
    train_result = classification_results(y_true=y_train, y_pred_proba=y_train_pred_proba, name="Train", threshold=threshold)
    val_result = classification_results(y_true=y_val, y_pred_proba=y_val_pred_proba, name="Validation", threshold=threshold)
    test_result = classification_results_test(y_pred_proba=y_test_pred_proba, threshold=threshold, name="Test")
    if print_overfit:
        val_overfit = round(train_result.loc['Train', 'roc_auc'] - val_result.loc['Validation', 'roc_auc'],3)
        test_overfit = round(train_result.loc['Train', 'roc_auc'] - test_result.loc['Test', 'roc_auc'],3)
        if val_overfit > 0.05 or test_overfit > 0.05:
            print("WARNING: High Overfit \nWe expect to be less than 0.05! Otherwise, there is a high chance that your model won't perform as good as you expected.")
        print("Overfit AUC (Train-Val):", val_overfit)
        print("Overfit AUC (Train-Test):", test_overfit)
    return pd.concat([train_result, val_result, test_result], axis=0)
