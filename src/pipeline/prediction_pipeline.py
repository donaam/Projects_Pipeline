import sys
import os
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_obj


class PredictPipeline:
    def __init__(self) -> None:
        pass

    def predict(self, features):
        try:
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            model_path = os.path.join('artifacts', 'model.pkl')

            preprocessor = load_obj(preprocessor_path)
            model = load_obj(model_path)

            data_scaled = preprocessor.transform(features)
            pred = model.predict(data_scaled)
            return pred

        except Exception as e:
            logging.info
            ('Error occurred in predict function of prediction pipeline')
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        cc_num: float,
        amt: float,
        zip: int,
        lat: float,
        long: float,
        city_pop: int,
        unix_time: int,
        merch_lat: float,
        merch_long: float,
        user_transaction_count: int,
        merchant_transaction_count: int,
        merchant: str,
        category: str,
        first: str,
        last: str,
        gender: str,
        street: str,
        city: str,
        state: str,
        job: str,
        dob: str,
        date: str,
        time: str
    ):
        self.cc_num = cc_num
        self.amt = amt
        self.zip = zip
        self.lat = lat
        self.long = long
        self.city_pop = city_pop
        self.unix_time = unix_time
        self.merch_lat = merch_lat
        self.merch_long = merch_long
        self.user_transaction_count = user_transaction_count
        self.merchant_transaction_count = merchant_transaction_count
        self.merchant = merchant
        self.category = category
        self.first = first
        self.last = last
        self.gender = gender
        self.street = street
        self.city = city
        self.state = state
        self.job = job
        self.dob = dob
        self.date = date
        self.time = time

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'cc_num': [self.cc_num],
                'amt': [self.amt],
                'zip': [self.zip],
                'lat': [self.lat],
                'long': [self.long],
                'city_pop': [self.city_pop],
                'unix_time': [self.unix_time],
                'merch_lat': [self.merch_lat],
                'merch_long': [self.merch_long],
                'user_transaction_count': [self.user_transaction_count],
                'merchant_transaction_count': [
                    self.merchant_transaction_count],
                'merchant': [self.merchant],
                'category': [self.category],
                'first': [self.first],
                'last': [self.last],
                'gender': [self.gender],
                'street': [self.street],
                'city': [self.city],
                'state': [self.state],
                'job': [self.job],
                'dob': [self.dob],
                'date': [self.date],
                'time': [self.time]
            }

            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Input DataFrame created successfully')
            return df

        except Exception as e:
            logging.info(
                'Error occurred in get_data_as_dataframe method of CustomData')
            raise CustomException(e, sys)
