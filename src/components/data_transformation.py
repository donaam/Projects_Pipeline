import os
import sys
import pandas as pd
from dataclasses import dataclass
from scipy.sparse import hstack
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from src.logger import logging
from src.exception import CustomException
from src.utils import save_function


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

        # Predefined columns
        self.numerical_cols = [
            'cc_num', 'amt', 'zip', 'lat', 'long', 'city_pop',
            'unix_time', 'merch_lat', 'merch_long',
            'user_transaction_count', 'merchant_transaction_count'
        ]
        self.categorical_cols = [
            'merchant', 'category', 'first', 'last', 'gender', 'street',
            'city', 'state', 'job', 'dob', 'date', 'time'
        ]

    def get_preprocessor(self, df: pd.DataFrame):
        try:
            valid_num = [
                col for col in self.numerical_cols if col in df.columns]
            valid_cat = [
                col for col in self.categorical_cols if col in df.columns]

            missing = list(set(self.numerical_cols + self.categorical_cols)
                           - set(valid_num + valid_cat))
            if missing:
                logging.warning(f"Missing columns ignored: {missing}")

            num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ])
            cat_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(
                    handle_unknown='ignore', sparse_output=True))
            ])

            return ColumnTransformer([
                ('num', num_pipeline, valid_num),
                ('cat', cat_pipeline, valid_cat)
            ])
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Train and test data loaded")

            target = 'is_fraud'
            drop_cols = [target, 'trans_num']

            X_train = train_df.drop(columns=drop_cols)
            y_train = train_df[target].values.reshape(-1, 1)

            X_test = test_df.drop(columns=drop_cols)
            y_test = test_df[target].values.reshape(-1, 1)

            preprocessor = self.get_preprocessor(train_df)

            X_train_trans = preprocessor.fit_transform(X_train)
            X_test_trans = preprocessor.transform(X_test)

            train_arr = hstack([X_train_trans, y_train])
            test_arr = hstack([X_test_trans, y_test])

            save_function(self.config.preprocessor_obj_file_path, preprocessor)
            logging.info("Preprocessor saved successfully")

            return train_arr, test_arr, self.config.preprocessor_obj_file_path

        except Exception as e:
            logging.info("Error during data transformation")
            raise CustomException(e, sys)
