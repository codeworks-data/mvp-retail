import pandas as pd
import numpy as np
import datetime
from sklearn.metrics import mean_squared_error 


###########
###########
##Function for part II. Create the dataframe  
###########
###########


def reformat_date(date):
    'Function used in order to have all date with the same format : dd-mm-yyyy'
    L = []
    n = 0
    for i in date:
        
        if i in ['0','1','2','3','4','5','6','7','8','9']:
            L.append(i)
            n = n+1
        else:
            if n==2:
                L.append('-')
                n=0
            elif n == 1:
                temp = L[-1]
                L[-1] = '0'
                L.append(temp)
                L.append('-')
                n=0
            else:
                print('error')
                
    return ''.join(L)



def remove_null_value(df):
    'Function to remove rows where they are null values in those 3 columns (DOB, city_code, Gender)'
    
    df = df[df['DOB'].notna()]
    df = df[df['city_code'].notna()]
    df = df[df['Gender'].notna()]
    return df



def date_to_year_from_now(date):
    'Function to convert a date into years'
    
    day = int(date[:2])
    mounth = int(date[3:5]) 
    year = int(date[6:10])
    
    t_start = datetime.datetime(year,mounth,day)    
    t_end = datetime.datetime.now()
    return int((t_end - t_start).total_seconds()/(3600*24*30))



def age_from_transaction(df):
    'Function creating a column with the age when the purchase occurs'
    
    df['DOB'] = pd.to_datetime(df.DOB, format='%d-%m-%Y') 
    df['tran_date'] = pd.to_datetime(df.tran_date, format='%d-%m-%Y')
    df['diff_trans__DOB'] = (df.tran_date-df.DOB)
    df['age_when_purchase'] = df['diff_trans__DOB'].map(lambda x:x.days)//360.25
    df.drop('diff_trans__DOB', axis= 1 , inplace = True)
    return df



def to_numerical(df):
    'Function which change to numerical value for all used columns,'
    'drop all unused columns, and do give us the month from purchase '
    
    # reformat date to have same format for all tran_date
    df['tran_date'] = df['tran_date'].apply(reformat_date)
    
    # adding a new column with the number of months between the purchase month and now
    df['month_from_purchase'] = df['tran_date'].apply(date_to_year_from_now)
    df = age_from_transaction(df)
    
    # create 2 dict and map them to have numerical value for gender and store
    dict_store = {val : i for i,val in enumerate(df.Store_type.unique())}
    dict_gender = {val : i for i,val in enumerate(df.Gender.unique())}
    df['store_id'] = df['Store_type'].map(dict_store)
    df['gender_id'] = df['Gender'].map(dict_gender)
    
    return df[['cust_id', 'month_from_purchase', 'store_id', 'age_when_purchase', 'gender_id', 'city_code']]



def standardize(df):
    'Function which put all value of the df between 0 and 1'
    
    return (df-df.min())/(df.max()-df.min())



###########
###########
## Function for part III. Relation between customer
###########
###########


def similarity_pearson(x, y):
    'Function which return the pearson correlation between x and y'
    
    import scipy.stats
    return scipy.stats.pearsonr(x, y)[0]   



def find_neighbours(sim, treshold):
    'Function which return a list to find closest neighbours according to a treshold'
    
    return [sim.index[i] for i, v in enumerate(sim) if v>=treshold]



def select_customer(index , cust_id, df ):
    'Function which return index and cust_id either if we enter the index or the cust_id'
    
    # if we want to retrieve the cust_id, we can use the index
    if type(index) == int:
        customer_index = index
        customer_id = df.iloc[index,0]
      
    # if we want to retrieve the index, we can use the cust_id
    else :
        customer_index = df[df.cust_id == cust_id].index[0]
        customer_id = cust_id
    
    return customer_index, customer_id



def give_info_neighbours(df_non_stand, sim_df, list_of_neighbours):
    'function which give us information about neighbours'
    
    neighbour_info = df_non_stand.loc[list_of_neighbours]
    neighbour_info['pearson_similarity'] = sim_df[list_of_neighbours].transpose()
    neighbour_info.sort_values(by=['pearson_similarity'], ascending = False, inplace = True)
    return neighbour_info


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



def meatrics_mse_rmse(df):
    'Function used to return meam square error and root mean square error'
    'for a dataframe for which we know the real value and the predicted one'
    
    # real values are in the column number 2 : 'real_rating_from_......' which is the last column
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
## Function for part V. Final Part : Doing prediction by combining Part III. and Part IV. 
###########
###########


def do_prediction(customer_index, customer_id, max_neighbours, treshold,df_stand,df_non_stand,Product_mapped,user_item_matrix, verbose): 
    'Function which give us for a customer index Two data frame:'
    ' - reco_unknown give us the result of recommend_item_from_uid if known variable is set to false'
    ' - reco_known give us the result of recommend_item_from_uid if known variable is set to true'
    
    # we take back the full data frame
    df_restrain = df_stand.copy()
    size = len(df_restrain)
    
    # we calculate the similarity for the specific customer against all other customer
    sim_mat = np.array([similarity_pearson(df_restrain.iloc[customer_index,:], df_restrain.iloc[j,:])for j in range(0,size)])
    sim_df = pd.DataFrame(data = sim_mat.reshape(1, size))

    # we find neighbours
    list_of_neighbours = find_neighbours(sim_df.iloc[0], treshold)
    
    if verbose == True:
        print(f'This len for list of neighbours for the cust_id : {customer_id} with a threshold of {treshold} : \n')
        print(f'is actually {len(list_of_neighbours)} but we will restrict is to {max_neighbours} \n')

    # we map them to obtain cust_id
    neighbour_info = give_info_neighbours(df_non_stand, sim_df, list_of_neighbours)
    list_similar_user_indices = neighbour_info['cust_id'].tolist()

    # we find prediction for unknown item
    reco_unknown = recommend_item_from_uid(list_similar_user_indices[0], list_similar_user_indices[1:],max_neighbours, user_item_matrix,Product_mapped, known= False, items=23)

    # we find prediction for known item
    reco_known = recommend_item_from_uid(list_similar_user_indices[0], list_similar_user_indices[1:],max_neighbours, user_item_matrix,Product_mapped, known= True, items=23)

    return reco_unknown, reco_known


###########
###########
###########
## END of Function_02
###########
###########
###########
