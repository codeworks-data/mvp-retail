from typing import List
import random
import pandas as pd


def fill_with_other_items_randomly(items_recommended: List, all_items_ids: List) -> List:
    """
    If complete the list of recommended items with the rest of the items at random
    :param items_recommended: list, list of items recommended
    :return: list, ordered list of all the items to be recommended
    """
    items_not_recommended = [item_id for item_id in all_items_ids if item_id not in items_recommended]
    # recommend other items randomly
    random_items_not_recommended = random.sample(items_not_recommended, len(items_not_recommended))
    return items_recommended + random_items_not_recommended


def get_sales_until_date(transactions: pd.DataFrame, max_date: str = '01-01-2012',
                         fill_na_with_zero: bool = True) -> pd.DataFrame:
    """
    return the sales data (index: customers, columns: items, quantity sold since day1),
    an equivalent to recommendation table, until a certain date
    :param transactions: pd.DataFrame, all transactions
    :param max_date: str, maximum date to consider
    :return: pd.DataFrame, sales table (number of items sold to each user until 'max_date')
    """
    # Quantity of items sold for each user until 'max_date'
    trasaction_until_date = transactions[transactions.tran_date < pd.to_datetime(max_date, format='%d-%m-%Y')]
    trasaction_grouped_by_customer_item = (
        trasaction_until_date.groupby(['cust_id', 'item_id'])[['Qty']].sum().reset_index()
    )
    # pivoted the last table (put items into columns)
    sales_table = (
        pd.pivot_table(
            trasaction_grouped_by_customer_item, index='cust_id', columns='item_id', values='Qty'
        )
    )
    sales_table = sales_table.fillna(0) if fill_na_with_zero else sales_table

    # Add new columns for items not sold yet
    for item_id in transactions.item_id.unique():
        if item_id not in sales_table.columns:
            sales_table[item_id] = 0 if fill_na_with_zero else sales_table
    sales_table.columns = sorted(sales_table.columns)
    return sales_table
