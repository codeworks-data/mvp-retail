import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import datetime
from sklearn.metrics import mean_squared_error 
from surprise.model_selection import cross_validate













###########
###########
## Function for part III. Relation between customer
###########
###########


def similarity_pearson(x, y):
    'function which return the pearson correlation between x and y'
    import scipy.stats
    return scipy.stats.pearsonr(x, y)[0]   

def find_neighbours(sim, threshold):
    'function which return a list to find closest neighbours according to a threshold'
    return [sim.index[i] for i, v in enumerate(sim) if v>=threshold]

def select_customer(index , cust_id, df ):
    'return index and cust_id either if we select the index or the cust_id'
    if type(index) == int:
        customer_index = index
        customer_id = df.iloc[index,0]
        
    else :
        customer_index = df[df.cust_id == cust_id].index[0]
        customer_id = cust_id
    
    #print(f'If the customer ID is {customer_index} so the cust_id is {customer_id} ') 
    return customer_index, customer_id


def give_info_neighbours(df_non_stand, sim_df, list_of_neighbours):
    'function which give us info on neighbours'
    neighbour_info = df_non_stand.loc[list_of_neighbours]
    neighbour_info['pearson_similarity'] = sim_df[list_of_neighbours].transpose()
    neighbour_info.sort_values(by=['pearson_similarity'], ascending = False, inplace = True)
    return neighbour_info





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




###########
###########
## Function for part V. Final Part : Doing prediction by combining Part III. and Part IV. 
###########
###########


def do_prediction(customer_index, customer_id, max_neighbours, threshold,df_stand,df_non_stand,Product_mapped,user_item_matrix): 
    #We take back the full data frame
    df_restrain = df_stand.copy()
    size = len(df_restrain)
    #we calculate the sim for the specific customer
    sim_mat = np.array([similarity_pearson(df_restrain.iloc[customer_index,:], df_restrain.iloc[j,:])for j in range(0,size)])
    sim_df = pd.DataFrame(data = sim_mat.reshape(1, size))

    #we find neighbours
    list_of_neighbours = find_neighbours(sim_df.iloc[0], threshold)


    #we map them to obtain cust_id
    neighbour_info = give_info_neighbours(df_non_stand, sim_df, list_of_neighbours)
    list_similar_user_indices = neighbour_info['cust_id'].tolist()
    #print('list_similar_user_indices :', list_similar_user_indices)

    #we find prediction for unknown item
    reco_unknown = recommend_item_for_all(list_similar_user_indices[0], list_similar_user_indices[1:],max_neighbours, user_item_matrix,Product_mapped, known= False)

    #we find prediction for known item
    reco_known = recommend_item_for_all(list_similar_user_indices[0], list_similar_user_indices[1:],max_neighbours, user_item_matrix,Product_mapped, known= True)

    return reco_unknown, reco_known







###############################





def compute_result(algor, my_k, my_min_k, my_sim_option, data):

    algo = algor(
        k = my_k, min_k = my_min_k, 
        sim_options = my_sim_option, verbose = True
        )

    results = cross_validate(
        algo = algo, data = data, measures=['RMSE'], 
        cv=5, return_train_measures=True
        )
    
 
    return algo, algor, my_k, my_min_k, results['test_rmse'].mean()




###############




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
        #information_prediction.drop('object_id', axis= 1, inplace = True)


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







###############


def prediction_with_treshold(user_id_to_predict, treshold, df):
    return df[(df['est']>treshold)&(df['uid']==user_id_to_predict)].sort_values(by=['est'], ascending=False)









####MERGING AALLLL 





def merge_for_known(pred_known_01,pred_known_02, pred_known_03):
    final_pred_known = pd.merge(left = pred_known_01,
                             right = pred_known_02,
                             left_on = "object_id",
                             right_on = "iid",
                             how = "left")

    final_pred_known = final_pred_known[['object_name','object_id','r_ui', 'mean_from_similar_user', 'est', 'details']]
    final_pred_known.rename({'mean_from_similar_user' : 'pred_01', 'r_ui' : 'real_rank', 'est': 'pred_02'}, axis= 1, inplace=True)

    final_pred_known.loc[final_pred_known['details'] != "{'was_impossible': True, 'reason': 'Not enough neighbors.'} " , 'pred02_worked'] = True 
    final_pred_known.loc[final_pred_known['details'] == "{'was_impossible': True, 'reason': 'Not enough neighbors.'} " , 'pred02_worked'] = False


    final_pred_known = pd.merge(left = final_pred_known,
                             right = pred_known_03,
                             left_on = "object_id",
                             right_on = "object_id",
                             how = "left")
    final_pred_known.rename({'mean_from_similar_user' : 'pred_03', 'object_name_x' : 'object_name' }, axis=1, inplace = True)
    final_pred_known.drop(['object_name_y','details', final_pred_known.columns[-1]],axis=1, inplace=True)
    return final_pred_known

def consolidate_know_function(pred_known_01,pred_known_02, pred_known_03, A,B,C):
    consolidated_known = merge_for_known(pred_known_01,pred_known_02, pred_known_03)
    consolidated_known['total'] = (A * consolidated_known['pred_01'] + B * consolidated_known['pred_02'] + C * consolidated_known['pred_03'])
    consolidated_known['div_by'] = np.nan
    consolidated_known['div_by'] = np.round(np.where(consolidated_known['pred_01'] > 0, consolidated_known['div_by'] + A, 0), 1)
    consolidated_known['div_by'] = np.round(np.where(consolidated_known['pred_02'] > 0, consolidated_known['div_by'] + B, 0), 1)
    consolidated_known['div_by'] = np.round(np.where(consolidated_known['pred_03'] > 0, consolidated_known['div_by'] + C, A+B), 1) #A+B par default mais surement faux
    consolidated_known[['div_by']] = consolidated_known[['div_by']].fillna(value=A+B+C)
    
    
    consolidated_known['final_estimation'] = consolidated_known['total'] / consolidated_known['div_by']
    return consolidated_known









def meatrics_mse_rmse_consolidated(df):
    'This function is used to return mse and rmse for a dataframe for which we know the real value'
    #Real values are in the column number 2 : 'real_rank_from_......'
    realVals = df['real_rank']

    #Predicted values are in the column number 1 : 'mean_from_similar_user'
    predictedVals = df[df.columns[-1]]

    #calucul of mean squared error
    mse = mean_squared_error(realVals, predictedVals)
    #calucul of root mean squared error
    rmse = mean_squared_error(realVals, predictedVals, squared = False)
    return mse, rmse






def merge_for_unknown(pred_unknown_01,pred_unknown_02, pred_unknown_03):
    final_pred_unknown = pd.merge(left = pred_unknown_01,
                             right = pred_unknown_02,
                             left_on = "object_id",
                             right_on = "iid",
                             how = "left")

    final_pred_unknown = final_pred_unknown[['object_name','object_id','r_ui', 'mean_from_similar_user', 'est', 'details']]
    final_pred_unknown.rename({'mean_from_similar_user' : 'pred_01', 'r_ui' : 'rank_by_default', 'est': 'pred_02'}, axis= 1, inplace=True)

    final_pred_unknown.loc[final_pred_unknown['details'] != "{'was_impossible': True, 'reason': 'Not enough neighbors.'} " , 'pred02_worked'] = True 
    final_pred_unknown.loc[final_pred_unknown['details'] == "{'was_impossible': True, 'reason': 'Not enough neighbors.'} " , 'pred02_worked'] = False


    final_pred_unknown = pd.merge(left = final_pred_unknown,
                             right = pred_unknown_03,
                             left_on = "object_id",
                             right_on = "object_id",
                             how = "left")
    final_pred_unknown.rename({'mean_from_similar_user' : 'pred_03', 'object_name_x' : 'object_name'}, axis=1, inplace = True)
    final_pred_unknown.drop(['object_name_y','details' ],axis=1, inplace=True)
    return final_pred_unknown



def consolidate_unknown_function(pred_unknown_01,pred_unknown_02, pred_unknown_03, A,B,C):
    consolidated_unknown = merge_for_unknown(pred_unknown_01,pred_unknown_02, pred_unknown_03)
    consolidated_unknown['total'] = (A * consolidated_unknown['pred_01'] + B * consolidated_unknown['pred_02'] + C * consolidated_unknown['pred_03'])
    consolidated_unknown['div_by'] = np.nan
    consolidated_unknown['div_by'] = np.round(np.where(consolidated_unknown['pred_01'] > 0, consolidated_unknown['div_by'] + A, 0), 1)
    consolidated_unknown['div_by'] = np.round(np.where(consolidated_unknown['pred_02'] > 0, consolidated_unknown['div_by'] + B, 0), 1)
    consolidated_unknown['div_by'] = np.round(np.where(consolidated_unknown['pred_03'] > 0, consolidated_unknown['div_by'] + C, A+B), 1) #A+B par default mais surement faux
    consolidated_unknown[['div_by']] = consolidated_unknown[['div_by']].fillna(value=A+B+C)
    
    
    consolidated_unknown['final_estimation'] = consolidated_unknown['total'] / consolidated_unknown['div_by']
    return consolidated_unknown









