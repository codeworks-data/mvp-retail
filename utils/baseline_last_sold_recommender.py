from typing import List
import pandas as pd


class LastSoldRecommender:
    def fit(self, transactions_df: pd.DataFrame) -> None:
        """
        Set as attributes the last sold items for know customers, and overall last sold items for unknown customers
        :param transactions_df: pd.DataFrame, data frame with dated transaction
        :return: None
        """
        self.known_users_last_sold_items = transactions_df.groupby('cust_id').apply(
            lambda df: (
                df
                    .sort_values('transaction_id', ascending=False)
                    .drop_duplicates(['item_id'], keep='first')
                    .item_id.tolist()
            )
        )
        self.unknown_users_last_sold_items = (
            transactions_df
                .sort_values('transaction_id', ascending=False)
                .drop_duplicates(['item_id'], keep='first')
                .item_id.tolist()
        )

    def predict(self, users_ids: List) -> pd.DataFrame:
        """

        :param users_ids: return the ordered (by date) list of items sold for each user
        :return: pd.Series of length  'len(users_ids)' of a list of last items sold
        """
        known_users_ids = [uid for uid in users_ids if uid in self.known_users_last_sold_items.index]
        unknown_users_ids = [uid for uid in users_ids if uid not in self.known_users_last_sold_items.index]

        known_users_last_sold_items = self.known_users_last_sold_items.loc[known_users_ids]
        unknown_users_last_sold_items = pd.Series(index=unknown_users_ids)
        for uid in unknown_users_ids:
            unknown_users_last_sold_items.loc[uid] = self.unknown_users_last_sold_items
        return pd.concat([known_users_last_sold_items, unknown_users_last_sold_items])