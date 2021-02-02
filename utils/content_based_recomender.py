import os
from typing import List
import pandas as pd
import numpy as np


class ContentBasedRecommender:
    def __init__(self, features: List) -> None:
        self.items_features = features
        self.users_features_normalized_df = pd.DataFrame(columns=self.items_features)
        self.unknown_user_features_normalized_df = pd.DataFrame(columns=self.items_features)

    def fit(self, sales: pd.DataFrame, items: pd.DataFrame) -> None:
        """
        Fit the recommender
        :param sales: pd.DataFrame of shape (n_users, n_items), containing closed sales
        :param items: pd.DataFrame of shape (n_items, n_features), containing the items features
        :return: None
        """
        self.items = items
        items_ids = sales.columns.tolist()

        items_features = items.loc[items_ids, self.items_features].values  # Shape: (n_items, n_features)
        current_sales = sales[items_ids].values  # Shape: (n_users, n_items)

        users_features = np.matmul(current_sales, items_features)  # Shape: (n_users, n_features)
        sum_users_features = users_features.sum(axis=1)  # Shape: (n_users,)
        broad_casted_sum_users_features = (  # Shape: (n_users, n_features)
            sum_users_features.reshape((-1, 1)).repeat(len(self.items_features), axis=1)
        )

        users_features_normalized = users_features/broad_casted_sum_users_features
        self.users_features_normalized_df = pd.DataFrame(
            users_features_normalized, index=sales.index, columns=self.items_features
        )
        self.unknown_user_features_normalized_df = pd.DataFrame(self.users_features_normalized_df.mean()).transpose()

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

    def save_fitted_model(self, model_folder: str) -> None:
        """
        Save a fitted model
        :param model_folder: str, name of the folder to save the model into
        :return: None
        """
        os.mkdir(model_folder)
        self.users_features_normalized_df.to_csv(model_folder+'/users_features_normalized_df.csv')
        self.unknown_user_features_normalized_df.to_csv(model_folder+'/unknown_user_features_normalized_df.csv')
        self.items.to_csv(model_folder+'/items.csv')

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
