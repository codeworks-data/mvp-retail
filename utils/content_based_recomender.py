import os
from typing import List
import pandas as pd
import numpy as np


class ContentBasedRecommender:
    def __init__(self, features_names: List) -> None:
        self.items_features_names = features_names
        self.users_features_normalized_df = pd.DataFrame(columns=self.items_features_names)
        self.unknown_user_features_normalized_df = pd.DataFrame(columns=self.items_features_names)

    def fit(self, users_items_sales: pd.DataFrame, items_features: pd.DataFrame) -> None:
        """
        Fit the recommender
        :param sales: pd.DataFrame of shape (n_users, n_items), containing closed sales
        :param items: pd.DataFrame of shape (n_items, n_features), containing the items features
        :return: None
        """
        self.items_features = items_features
        items_ids = users_items_sales.columns.tolist()

        items_features_array = items_features.loc[items_ids, self.items_features_names].values  # Shape: (n_items, n_features)
        users_items_sales_array = users_items_sales[items_ids].values  # Shape: (n_users, n_items)

        users_features_preference = np.matmul(users_items_sales_array, items_features_array)  # Shape: (n_users, n_features)
        sum_users_features_preference = users_features_preference.sum(axis=1)  # Shape: (n_users,)
        broad_casted_sum_users_features_preference = (  # Shape: (n_users, n_features)
            sum_users_features_preference.reshape((-1, 1)).repeat(len(self.items_features_names), axis=1)
        )

        users_features_preference_normalized = users_features_preference/broad_casted_sum_users_features_preference
        self.known_users_features_preference_normalized_df = pd.DataFrame(
            users_features_preference_normalized, index=users_items_sales.index, columns=self.items_features_names
        )
        self.unknown_users_features_preference_normalized_df = pd.DataFrame(self.known_users_features_preference_normalized_df.mean()).transpose()

    def predict(self, users_ids: List) -> pd.DataFrame:
        """
        Predict categories to be sold the next month for the users lise
        :param users_ids: List[int] users IDs
        :return: pd.Dataframe, if shape (len(users_ids), n_items), probabilities of sale
        """
        known_users_ids = [uid for uid in users_ids if uid in self.known_users_features_preference_normalized_df.index]
        unknown_users_ids = [uid for uid in users_ids if
                             uid not in self.known_users_features_preference_normalized_df.index]
        n_unknown_users = len(unknown_users_ids)

        known_users_features_preference_array = self.known_users_features_preference_normalized_df.loc[
            known_users_ids].values
        unknown_users_features_preference_array = self.unknown_users_features_preference_normalized_df.values.repeat(
            n_unknown_users, axis=0)

        users_features_preference_array = np.concatenate([
            known_users_features_preference_array, unknown_users_features_preference_array
        ])

        users_items_relevence = pd.DataFrame(
            np.matmul(users_features_preference_array, self.items_features.transpose().values),
            index=known_users_ids + unknown_users_ids, columns=self.items_features.index
        )

        return users_items_relevence

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
