import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import operator
from sklearn.metrics import mean_squared_error 


###########
###########
## Function for part III. Selecting similar users
###########
###########


def similar_users(user_id, matrix, k=6):
    'Function which return a list of k user from the matrix which are the most similar to user_id with the cosine similarity'
    
    # create an array of just the current user
    user = matrix[matrix.index == user_id]
    
    # and an array with all other users
    other_users = matrix[matrix.index != user_id]
    
    # calc cosine similarity between user and each other user
    similarities = cosine_similarity(user,other_users)[0].tolist()
    
    # create list of indices of these users
    indices = other_users.index.tolist()
    
    # create key/values pairs of user index and their similarity
    index_similarity = dict(zip(indices, similarities))
    
    # sort by similarity
    index_similarity_sorted = sorted(index_similarity.items(), key=operator.itemgetter(1))
    index_similarity_sorted.reverse()
    
    # take k users from the top
    top_users_similarities = index_similarity_sorted[:k]
    users = [u[0] for u in top_users_similarities]
    
    return users
    

###########
###########
## Function for part IV. Predictions on item
###########
###########


def recommend_item_from_uid(user_index, similar_user_indices,max_value, matrix, Product_map, known, items=23):
    'This function allow us to do recommendation either on :'
    ' - known item if "known" variable is set to True'
    ' - unknown if it is set to False'
    'This function is a bit modify but inspired by :'
    'https://towardsdatascience.com/build-a-user-based-collaborative-filtering-recommendation-engine-for-anime-92d35921f304'
    
    similar_user_indices = similar_user_indices[:max_value]
    
    # load vectors for similar users
    similar_users = matrix[matrix.index.isin(similar_user_indices)]

    # calculate average ratings across the 3 similar users and only on non zero value and fill with 0 after
    # this allow us not to take zero values into account in the average
    similar_users = similar_users.replace(0, np.NaN).mean(axis=0).replace(np.NaN, 0)
    
    # convert to dataframe so its easy to sort and filter
    similar_users_df = pd.DataFrame(similar_users, columns=['mean'])
    
    # load vector for the current user
    user_df = matrix[matrix.index == user_index]
    
    # transpose it so its easier to filter
    user_df_transposed = user_df.transpose()
    
    # rename the column as 'rating'
    user_df_transposed.columns = ['rating']
    
    if known == True:
        # remove any rows without a 0 value. Item with no rating yet
        user_df_transposed = user_df_transposed[user_df_transposed['rating']!=0]
    else:
        # remove any rows without a 0 value. Item with no rating yet
        user_df_transposed = user_df_transposed[user_df_transposed['rating']==0]
        
    # generate a list of items the user has not rated
    items_seen = user_df_transposed.index.tolist()
    
    # filter avg ratings of similar users for only items the current user has not rated
    similar_users_df_filtered = similar_users_df[similar_users_df.index.isin(items_seen)]
    
    # order the dataframe by mean values
    similar_users_df_ordered = similar_users_df_filtered.sort_values(by=['mean'], ascending=False)
    
    # grab the top n item  
    top_n_item = similar_users_df_ordered.head(items)
    top_n_item_indices = top_n_item.index.tolist()

    # lookup these object in the other dataframe to find names
    P_H = Product_map[['object_name', 'object_id']]
    top_object = P_H[P_H['object_id'].isin(top_n_item_indices)]
    


    # do an inner join over index and the object id
    information_prediction = pd.merge(left = top_object,
                         right = similar_users_df_ordered,
                         left_on = "object_id",
                         right_index=True
                   )
    
    # order by mean value ascending
    information_prediction.sort_values(by=['mean'], ascending=False, inplace=True)
    
    # rename la colone
    information_prediction.rename({'mean': 'mean_from_similar_user'}, axis=1, inplace = True)
    
    if known == True:
    
        # do another merge to check the esimation with the real value
        information_prediction = pd.merge(left = information_prediction,
                             right = matrix.loc[user_index].reset_index(),
                             left_on = "object_id",
                             right_on = 'object_id'
                       )


        # rename la column with user id and the real value
        information_prediction.rename({user_index: 'real_rating_from_'+str(user_index)}, axis=1, inplace = True)
    
    
    
    return information_prediction 


###########
###########
## Function for part V. Metrics
###########
###########


def meatrics_mse_rmse(df):
    'Function used to return meam square error and root mean square error'
    'for a dataframe for which we know the real value and the predicted one'
    
    # real values are in the column number 2 : 'real_rating_from_......' which is also the last column
    realVals = df[df.columns[-1]]
    
    # predicted values are in the column number 1 : 'mean_from_similar_user'
    predictedVals = df['mean_from_similar_user']
    
    # calucul of mean squared error
    mse = mean_squared_error(realVals, predictedVals)
    
    # calucul of root mean squared error
    rmse = mean_squared_error(realVals, predictedVals, squared = False)
    
    return mse, rmse


###########
###########
###########
## END of Function_04
###########
###########
###########
