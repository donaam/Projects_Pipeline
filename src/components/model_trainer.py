import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from scipy.sparse import issparse

from src.logger import logging
from src.exception import CustomException
from src.utils import save_function, classification_model_performance


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info("Splitting features and target variables")

            # Convert to CSR if sparse to allow slicing
            if issparse(train_array):
                train_array = train_array.tocsr()
            if issparse(test_array):
                test_array = test_array.tocsr()

            X_train = train_array[:, :-1]
            y_train = train_array[:, -1].toarray().ravel()
            X_test = test_array[:, :-1]
            y_test = test_array[:, -1].toarray().ravel()

            models = {
                'LogisticRegression': LogisticRegression(max_iter=1000),
                'RandomForestClassifier': RandomForestClassifier(),
                'DecisionTreeClassifier': DecisionTreeClassifier(),
            }

            report = classification_model_performance(
                X_train, y_train, X_test, y_test, models)

            print("\nModel Evaluation Report:")
            for name, metrics in report.items():
                print(f"{name}: {metrics}")
            logging.info(f"Model Report: {report}")

            best_model_name = max(report, key=lambda x: report[x]['f1_score'])
            best_model = models[best_model_name]
            best_f1 = report[best_model_name]['f1_score']

            print(f"\nBest Model: {best_model_name} (F1 Score: {best_f1})")
            logging.info(
                f"Best Model: {best_model_name} (F1 Score: {best_f1})")

            save_function(self.config.trained_model_file_path, best_model)
            logging.info("Best model saved successfully.")

        except Exception as e:
            logging.error("Error during model training")
            raise CustomException(e, sys)
