import os
import sys
import pickle
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score)
from src.exception import CustomException
from src.logger import logging


def save_function(file_path, obj):
    """
    Saves a Python object to disk using pickle.
    """
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)
    with open(file_path, "wb") as file_obj:
        pickle.dump(obj, file_obj)


def load_obj(file_path):
    """
    Loads a pickled Python object from disk.
    """
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Error in load_object function in utils')
        raise CustomException(e, sys)


def classification_model_performance(X_train, y_train, X_test, y_test, models):
    """
    Trains and evaluates multiple classification models.
    Returns a report dictionary with accuracy, precision, recall, and f1_score.
    """
    try:
        report = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            report[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1_score': f1_score(y_test, y_pred, zero_division=0)
            }
        return report

    except Exception as e:
        raise CustomException(e, sys)
