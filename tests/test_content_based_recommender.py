import unittest
import pandas as pd

from utils.content_based_recomender import ContentBasedRecommender


class TestContentBasedRecommender(unittest.TestCase):
    def test_fit_user_features_should_be_of_shape_n_user_n_features(self):
        # GIVEN
        features_ids = ['feat_1', 'feat_2']
        products_ids = ['prod_1', 'prod_2', 'prod_3']
        customer_ids = ['cust_1', 'cust_2', 'cust_3']
        items = pd.DataFrame(
            [[0, 1], [1, 0], [1, 1]],
            index=products_ids, columns=features_ids
        )
        sales = pd.DataFrame(
            [[0, 1, 1], [1, 0, 1], [1, 1, 1]],
            index=customer_ids, columns=products_ids
        )
        expected_first_dimension = len(customer_ids)
        expected_second_dimension = len(features_ids)

        cbr = ContentBasedRecommender(features_ids)

        # WHEN
        cbr.fit(sales, items)

        # THEN
        returned_first_dimension, returned_second_dimension = cbr.users_features_normalized_df.shape
        self.assertEqual(
            returned_first_dimension, expected_first_dimension,
            'User features table does not contain {}(number of users) rows'.format(expected_first_dimension)
        )
        self.assertEqual(
            returned_second_dimension, expected_second_dimension,
            'User features table does not contain {}(number of features) columns'.format(expected_first_dimension)
        )

    def test_fit_unknown_user_features_should_be_of_shape_1_n_features(self):
        # GIVEN
        features_ids = ['feat_1', 'feat_2']
        products_ids = ['prod_1', 'prod_2', 'prod_3']
        customer_ids = ['cust_1', 'cust_2', 'cust_3']
        items = pd.DataFrame(
            [[0, 1], [1, 0], [1, 1]],
            index=products_ids, columns=features_ids
        )
        sales = pd.DataFrame(
            [[0, 1, 1], [1, 0, 1], [1, 1, 1]],
            index=customer_ids, columns=products_ids
        )
        expected_first_dimension = 1
        expected_second_dimension = len(features_ids)

        cbr = ContentBasedRecommender(features_ids)

        # WHEN
        cbr.fit(sales, items)

        # THEN
        returned_first_dimension, returned_second_dimension = cbr.unknown_user_features_normalized_df.shape
        self.assertEqual(
            returned_first_dimension, expected_first_dimension,
            'User features table does not contain 1 row'
        )
        self.assertEqual(
            returned_second_dimension, expected_second_dimension,
            'User features table does not contain {}(number of features) columns'.format(expected_first_dimension)
        )

    def test_fit_sample_user_features_should_be_correct(self):
        # GIVEN
        features_ids = ['feat_1', 'feat_2']
        products_ids = ['prod_1', 'prod_2', 'prod_3']
        customer_ids = ['cust_1', 'cust_2', 'cust_3']
        items = pd.DataFrame(
            [[0, 1], [1, 0], [1, 1]],
            index=products_ids, columns=features_ids
        )
        sales = pd.DataFrame(
            [[0, 1, 1], [1, 0, 1], [1, 1, 1]],
            index=customer_ids, columns=products_ids
        )
        expected_users_features_normalized_df = pd.DataFrame(
            [[2 / 3, 1 / 3], [1 / 3, 2 / 3], [1 / 2, 1 / 2]],
            index=customer_ids, columns=features_ids
        )

        cbr = ContentBasedRecommender(features_ids)

        # WHEN
        cbr.fit(sales, items)

        # THEN
        returned_users_features_normalized_df = cbr.users_features_normalized_df
        self.assertTrue(returned_users_features_normalized_df.equals(expected_users_features_normalized_df))

    def test_fit_user_features_should_be_of_shape_n_user_n_items(self):
        # GIVEN
        features_ids = ['feat_1', 'feat_2']
        products_ids = ['prod_1', 'prod_2', 'prod_3']
        customer_ids = ['cust_1', 'cust_2', 'cust_3']
        prediction_customer_ids = ['cust_2', 'cust_4']
        items = pd.DataFrame(
            [[0, 1], [1, 0], [1, 1]],
            index=products_ids, columns=features_ids
        )
        sales = pd.DataFrame(
            [[0, 1, 1], [1, 0, 1], [1, 1, 1]],
            index=customer_ids, columns=products_ids
        )
        expected_first_dimension = len(prediction_customer_ids)
        expected_second_dimension = len(products_ids)

        cbr = ContentBasedRecommender(features_ids)

        # WHEN
        cbr.fit(sales, items)

        # THEN
        returned_first_dimension, returned_second_dimension = cbr.predict(prediction_customer_ids).shape
        self.assertEqual(
            returned_first_dimension, expected_first_dimension,
            'User features table does not contain {}(number of users) rows'.format(expected_first_dimension)
        )
        self.assertEqual(
            returned_second_dimension, expected_second_dimension,
            'User features table does not contain {}(number of features) columns'.format(expected_first_dimension)
        )

    def test_predict_recommendation_table_should_be_of_shape_len_users_ids_n_items(self):
        # GIVEN
        features_ids = ['feat_1', 'feat_2']
        products_ids = ['prod_1', 'prod_2', 'prod_3']
        customer_ids = ['cust_1', 'cust_2', 'cust_3']
        prediction_customer_ids = ['cust_2', 'cust_4']
        items = pd.DataFrame(
            [[0, 1], [1, 0], [1, 1]],
            index=products_ids, columns=features_ids
        )
        expected_first_dimension = len(prediction_customer_ids)
        expected_second_dimension = len(products_ids)

        cbr = ContentBasedRecommender(features_ids)
        cbr.users_features_normalized_df = pd.DataFrame(
            [[0, 1], [1, 0], [1, 1]], columns=features_ids, index=customer_ids
        )
        cbr.unknown_user_features_normalized_df = pd.DataFrame([[.5, .5]], columns=features_ids)
        cbr.items = items

        # WHEN
        predictions = cbr.predict(prediction_customer_ids)

        # THEN
        returned_first_dimension, returned_second_dimension = predictions.shape
        self.assertEqual(
            returned_first_dimension, expected_first_dimension,
            'User features table does not contain {}(number of users) rows'.format(expected_first_dimension)
        )
        self.assertEqual(
            returned_second_dimension, expected_second_dimension,
            'User features table does not contain {}(number of features) columns'.format(expected_first_dimension)
        )

    def test_predict_recommendation_table_should_be_of_shape_len_users_ids_n_items(self):
        # GIVEN
        features_ids = ['feat_1', 'feat_2']
        products_ids = ['prod_1', 'prod_2', 'prod_3']
        customer_ids = ['cust_1', 'cust_2', 'cust_3']
        prediction_customer_ids = ['cust_3', 'cust_4']
        items = pd.DataFrame(
            [[0, 1], [1, 0], [1, 1]],
            index=products_ids,
            columns=features_ids
        )

        cbr = ContentBasedRecommender(features_ids)
        cbr.items_features = features_ids
        cbr.users_features_normalized_df = pd.DataFrame(
            [[1, 0], [0, 1], [.6, .4]], columns=features_ids, index=customer_ids
        )
        cbr.unknown_user_features_normalized_df = pd.DataFrame([[.2, .8]], columns=features_ids)
        cbr.items = items

        expected_predictions = pd.DataFrame(
            [[.4, .6, 1.], [.8, .2, 1.]],
            columns=products_ids, index=prediction_customer_ids
        )

        # WHEN
        predictions = cbr.predict(prediction_customer_ids)

        # THEN
        self.assertTrue(expected_predictions.equals(predictions), 'Predictions are not correct')
