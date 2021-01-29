import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import operator
import statistics
from sklearn.metrics import mean_squared_error 


###########
###########
## Function for part III. Selecting similar users
###########
###########


def similar_users(user_id, matrix, k=6):
    'This function return a list of k user from the matrix which are the most similar to user_id with cosine similarity'
    
    # create a df of just the current user
    user = matrix[matrix.index == user_id]
    
    # and a df of all other users
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
    
    # grab k users off the top
    top_users_similarities = index_similarity_sorted[:k]
    users = [u[0] for u in top_users_similarities]
    
    return users
    

###########
###########
## Function for part IV. Predictions on item
###########
###########


def recommend_item_for_all(user_index, similar_user_indices,max_value, matrix,Product_map, known, items=23):
    'This function allow us to do recommendation either on known item if known is set to True or unknown if it is set to False'
    'This function a bit adapted is taken from this article :'
    'https://towardsdatascience.com/build-a-user-based-collaborative-filtering-recommendation-engine-for-anime-92d35921f304 '
    
    similar_user_indices = similar_user_indices[:max_value]
    
    # load vectors for similar users
    similar_users = matrix[matrix.index.isin(similar_user_indices)]

    # calc avg ratings across the 3 similar users only on non 0 value and retrieve 0 after
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
        # remove any rows without a 0 value. Anime not watched yet
        user_df_transposed = user_df_transposed[user_df_transposed['rating']!=0]
    else:
        # remove any rows without a 0 value. Anime not watched yet
        user_df_transposed = user_df_transposed[user_df_transposed['rating']==0]
        
    # generate a list of animes the user has not seen
    animes_seen = user_df_transposed.index.tolist()
    
    # filter avg ratings of similar users for only anime the current user has not seen
    similar_users_df_filtered = similar_users_df[similar_users_df.index.isin(animes_seen)]
    
    # order the dataframe
    similar_users_df_ordered = similar_users_df_filtered.sort_values(by=['mean'], ascending=False)
    # grab the top n anime   
    top_n_anime = similar_users_df_ordered.head(items)
    top_n_anime_indices = top_n_anime.index.tolist()

    # lookup these object in the other dataframe to find names
    P_H = Product_map[['object_name', 'object_id']]
    top_object = P_H[P_H['object_id'].isin(top_n_anime_indices)]
    


    #do an inner join over index and the object id
    information_prediction = pd.merge(left = top_object,
                         right = similar_users_df_ordered,
                         left_on = "object_id",
                         right_index=True
                   )
    
    #order by mean value ascending
    information_prediction.sort_values(by=['mean'], ascending=False, inplace=True)
    
    #rename la colone
    information_prediction.rename({'mean': 'mean_from_similar_user'}, axis=1, inplace = True)
    
    if known == True:
    
        #do another merge to check with previous value
        information_prediction = pd.merge(left = information_prediction,
                             right = matrix.loc[user_index].reset_index(),
                             left_on = "object_id",
                             right_on = 'object_id'
                       )


        #on drop la colone object qui apparait deux fois
        information_prediction.drop('object_id', axis= 1, inplace = True)


        #rename la column with user id and the real value
        information_prediction.rename({user_index: 'real_rank_from_'+str(user_index)}, axis=1, inplace = True)
    
    
    
    return information_prediction #items


###########
###########
## Function for part V. Metrics
###########
###########


def meatrics_mse_rmse(df):
    
    'This function is used to return mse and rmse for a dataframe for which we know the real value'
    
    #Real values are in the column number 2 : 'real_rank_from_......'
    realVals = df[df.columns[2]]
    #Predicted values are in the column number 1 : 'mean_from_similar_user'
    predictedVals = df[df.columns[1]]
    #calucul of mean squared error
    mse = mean_squared_error(realVals, predictedVals)
    #calucul of root mean squared error
    rmse = mean_squared_error(realVals, predictedVals, squared = False)
    return mse, rmse






