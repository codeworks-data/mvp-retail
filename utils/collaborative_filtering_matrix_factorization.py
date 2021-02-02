import os
from typing import List
import pandas as pd
import numpy as np
import joblib
import implicit
from scipy.sparse import coo_matrix


class MFRecommender:
    def __init__(self, **params) -> None:
        self.model = implicit.als.AlternatingLeastSquares(**params)

    def fit(self, sales: pd.DataFrame) -> None:
        """
        Fit the recommender
        :param sales: pd.DataFrame of shape (n_users, n_items), containing closed sales
        :return: None
        """
        # replace ids with indices
        sales_with_ids_stripped = sales.loc[sorted(sales.index), sorted(sales.columns)].copy()
        new_users_ids = list(range(len(sales_with_ids_stripped)))
        new_items_ids = list(range(len(sales_with_ids_stripped.columns)))
        self.items_dict = {x: y for x, y in zip(sales_with_ids_stripped.columns, new_items_ids)}
        # self.items_reverse_dict = {x:y for y,x in zip(sorted(sales.columns),new_items_ids)}
        self.users_dict = {x: y for x, y in zip(sales_with_ids_stripped.index, new_users_ids)}
        sales_with_ids_stripped.index = new_users_ids
        sales_with_ids_stripped.columns = new_items_ids

        # transform sales df to coo_matrix
        CUTOMER_COLUMN_NAME = 'cust_id'
        ITEM_COLUMN_NAME = 'item_id'
        VALUE_COLUMN_NAME = 'quantity'
        sales_with_ids_stripped.index.name = CUTOMER_COLUMN_NAME
        sales_with_ids_stripped_melted = pd.melt(
            sales_with_ids_stripped.reset_index(),
            id_vars=CUTOMER_COLUMN_NAME,
            value_vars=new_items_ids, var_name=ITEM_COLUMN_NAME,
            value_name=VALUE_COLUMN_NAME
        ).dropna(subset=[VALUE_COLUMN_NAME])
        item_user_data = coo_matrix(
            (sales_with_ids_stripped_melted[VALUE_COLUMN_NAME],
            (sales_with_ids_stripped_melted[ITEM_COLUMN_NAME],
            sales_with_ids_stripped_melted[CUTOMER_COLUMN_NAME])),
            shape=(len(new_items_ids), len(new_users_ids))
        )

        # fit the model
        self.model.fit(item_user_data)

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

    def save_fitted_model(self, model_folder: str) -> None:
        """
        Save a fitted model
        :param model_folder: str, name of the folder to save the model into
        :return: None
        """
        os.mkdir(model_folder)
        joblib.dump(self.model, model_folder + '/als_model.joblib')
        joblib.dump(self.items_dict, model_folder + '/items_dict.joblib')
        joblib.dump(self.users_dict, model_folder + '/users_dict.joblib')

    def load_fitted_model(self, model_folder: str) -> None:
        """
        Load a fitted model
        :param model_folder: str, name of the folder to load the model from
        :return: None
        """
        self.model = joblib.load(model_folder + '/als_model.joblib')
        self.items_dict = joblib.load(model_folder + '/items_dict.joblib')
        self.users_dict = joblib.load(model_folder + '/users_dict.joblib')
