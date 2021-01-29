import os
from typing import List
import pandas as pd
import numpy as np
import joblib
import implicit
from google.cloud import storage


class MFRecommender:
    def __init__(self, **params) -> None:
        self.model = implicit.als.AlternatingLeastSquares(**params)

    def predict(self, users_ids: List) -> pd.DataFrame:
        """
        Predict categories to be sold the next month for the users lise
        :param users_ids: List[int] users IDs
        :return: pd.Dataframe, if shape (len(users_ids), n_items), probabilities of sale
        """
        known_users_ids = [uid for uid in users_ids if uid in self.users_dict]
        known_users_indices = [self.users_dict[uid] for uid in known_users_ids]
        unknown_users_ids = [uid for uid in users_ids if uid not in self.users_dict]
        n_unknown_users = len(unknown_users_ids)

        known_users_features_array = self.model.user_factors[known_users_indices, :]
        unknown_users_features_array = self.model.user_factors.mean(axis=0).reshape(
            (-1, self.model.factors)).repeat(n_unknown_users, axis=0)

        users_features_array = np.concatenate([known_users_features_array, unknown_users_features_array])

        return pd.DataFrame(
            np.matmul(users_features_array, self.model.item_factors.T),
            index=known_users_ids + unknown_users_ids,
            columns=[k for k, v in sorted(self.items_dict.items(), key=lambda item: item[1])]
        )

    def _load_joblib(self, file_name):

        storage_client = storage.Client()
        bucket_name = os.environ['MODEL_BUCKET']
        model_item = os.environ['MODEL_DIR']+'/'+file_name
        model_local = '/tmp/local.joblib'

        bucket = storage_client.get_bucket(bucket_name)
        # select bucket file
        blob = bucket.blob(model_item)
        # download that file and name it 'local.joblib'
        blob.download_to_filename(model_local)
        # load that file from local file
        job = joblib.load(model_local)
        return job

    def load_fitted_model(self) -> None:
        """
        Load a fitted model
        :param model_folder: str, name of the folder to load the model from
        :return: None
        """
        self.model = self._load_joblib('als_model.joblib')
        self.items_dict = self._load_joblib('items_dict.joblib')
        self.users_dict = self._load_joblib('users_dict.joblib')


def cloud_predict(request):
    mfr = MFRecommender()
    mfr.load_fitted_model()
    request_json = request.get_json()

    if request.args and 'message' in request.args:
        ids = request.args.get('message')
    elif request_json and 'message' in request_json:
        ids = request_json['message']
    else:
        return f'Fail!'

    return mfr.predict(ids).to_dict()
