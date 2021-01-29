import os
from typing import List

import pandas as pd
import numpy as np


class ContentBasedRecommender:
    def __init__(self, features: List) -> None:
        self.items_features = features
        self.users_features_normalized_df = pd.DataFrame(columns=self.items_features)
        self.unknown_user_features_normalized_df = pd.DataFrame(columns=self.items_features)

    def predict(self, users_ids: List) -> pd.DataFrame:
        """
        Predict categories to be sold the next month for the users lise
        :param users_ids: List[int] users IDs
        :return: pd.Dataframe, if shape (len(users_ids), n_items), probabilities of sale
        """
        known_users_ids = [uid for uid in users_ids if uid in self.users_features_normalized_df.index]
        unknown_users_ids = [uid for uid in users_ids if uid not in self.users_features_normalized_df.index]
        n_unknown_users = len(unknown_users_ids)

        known_users_features_array = self.users_features_normalized_df.loc[known_users_ids].values
        unknown_users_features_array = self.unknown_user_features_normalized_df.values.repeat(n_unknown_users, axis=0)

        users_features_array = np.concatenate([known_users_features_array, unknown_users_features_array])

        return pd.DataFrame(
            np.matmul(users_features_array, self.items.transpose().values),
            index=known_users_ids+unknown_users_ids, columns=self.items.index
        )

    def load_fitted_model(self, model_folder: str) -> None:
        """
        Load a fitted model
        :param model_folder: str, name of the folder to load the model from
        :return: None
        """
        self.users_features_normalized_df = pd.read_csv(model_folder+'/users_features_normalized_df.csv', index_col=0)
        self.unknown_user_features_normalized_df = pd.read_csv(model_folder+'/unknown_user_features_normalized_df.csv', index_col=0)
        self.items = pd.read_csv(model_folder+'/items.csv', index_col=0)
        self.items_features = self.users_features_normalized_df.columns.tolist()


def cloud_predict(request):
    cbr = ContentBasedRecommender([])
    cbr.load_fitted_model('gs://'+os.environment['MODEL_BUCKET']+'/'+os.environment['MODEL_DIR'])
    request_json = request.get_json()

    if request.args and 'message' in request.args:
        ids = request.args.get('message')
    elif request_json and 'message' in request_json:
        ids = request_json['message']
    else:
        return f'Fail!'

    return cbr.predict(ids).to_dict()

