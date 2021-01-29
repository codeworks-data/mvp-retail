from typing import List
import pandas as pd


class MostSoldRecommender:
    def fit(self, cumulative_sales: pd.DataFrame) -> None:
        """
        Set as attributes the most sold items for know customers, and overall most sold items for unknown customers
        :param transactions_df: cumulative quantity of items sold
        :return: None
        """
        self.known_users_most_sold_items = (
            cumulative_sales.apply(
                lambda s: s[s > 0].sort_values(ascending=False).index.tolist()
                , axis=1
            )
        )
        self.unknown_users_most_sold_items = cumulative_sales.sum().sort_values(ascending=False).index.tolist()

    def predict(self, users_ids: List) -> pd.DataFrame:
        """

        :param users_ids: return the ordered (by quantity) list of items sold for each user
        :return: pd.Series of length  'len(users_ids)' of a list of last items sold
        """
        known_users_ids = [uid for uid in users_ids if uid in self.known_users_most_sold_items.index]
        unknown_users_ids = [uid for uid in users_ids if uid not in self.known_users_most_sold_items.index]

        known_users_most_sold_items = self.known_users_most_sold_items.loc[known_users_ids]
        unknown_users_most_sold_items = pd.Series(index=unknown_users_ids)
        for uid in unknown_users_ids:
            unknown_users_most_sold_items.loc[uid] = self.unknown_users_most_sold_items
        return pd.concat([known_users_most_sold_items, unknown_users_most_sold_items])
