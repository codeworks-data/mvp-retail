import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error 

###########
###########
## Function for part V. Create the consolidated Data frame
###########
###########


def merge_pred(pred_01,pred_02, pred_03, known):
    'Function which merge all known prediction for a specific user'
    'We can see the orignal rate r_ui'
    'But also all predicted ratings : pred_01, pred_02, pred_03'

    # we merge pred_01 and pred_02 based on the object_id (iid)
    final_pred = pd.merge(left = pred_01,
                             right = pred_02,
                             left_on = "object_id",
                             right_on = "iid",
                             how = "left")

    #selecting and renaming some columns
    final_pred = final_pred[['object_name','object_id','r_ui', 'mean_from_similar_user', 'est', 'details']]
    final_pred.rename({'mean_from_similar_user' : 'pred_01', 'est': 'pred_02'}, axis = 1, inplace = True)
    
    # we change name if we know the actual rate
    if known == True:
        final_pred.rename({'r_ui' : 'real_rank'}, axis = 1, inplace = True)
    else:
        final_pred.rename({'r_ui' : 'rank_by_default'}, axis = 1, inplace = True)

    #give us information if we had enough neighbors to dio the prediction
    final_pred.loc[final_pred['details'] != "{'was_impossible': True, 'reason': 'Not enough neighbors.'} " , 'pred02_worked'] = True 
    final_pred.loc[final_pred['details'] == "{'was_impossible': True, 'reason': 'Not enough neighbors.'} " , 'pred02_worked'] = False

    # we merge pred_01/pred_02 and pred_03 based on the object_id
    final_pred = pd.merge(left = final_pred,
                             right = pred_03,
                             left_on = "object_id",
                             right_on = "object_id",
                             how = "left")
    
    # renaming and dropping some columns
    final_pred.rename({'mean_from_similar_user' : 'pred_03', 'object_name_x' : 'object_name' }, axis=1, inplace = True)
    final_pred.drop(['object_name_y','details'],axis=1, inplace=True)
    
    if known == True:
        return final_pred[['object_name', 'object_id', 'real_rank', 'pred_01', 'pred_02', 'pred02_worked', 'pred_03']]
    else:
        return final_pred[['object_name', 'object_id', 'rank_by_default', 'pred_01', 'pred_02', 'pred02_worked', 'pred_03']]



def consolidate_function(pred_01,pred_02, pred_03, A,B,C, known):
    'Function which return the consolidated data frame according to the weights values'
    'First, we merge all df thanks to merge_for_known'
    'Then, we do a weighted arithmetic mean with coefficient A, B, C'
    'So we use the total column to add value with related coefficient'
    'And used div_by to known where are non null value in order to divided by the good number'
    'For example : with A, B, C = (2,1,4)'
    'And pred_01 , pred_02, pred_03 = 2, 0, 4'
    'We have (pred_01x2 + pred_03x4) / (A + C) = 2.655, so we have to exclude pred_02 in the calculus'
    'We exclude prediction at 0, this mean the prediction did not work due to a lack of information (neighbors or rating)'
    
    # we do the merge for all data frame with prediction 
    consolidated = merge_pred(pred_01,pred_02, pred_03, known)
    
    # we do the total using coefficient A, B, C
    consolidated['total'] = (A * consolidated['pred_01'] + B * consolidated['pred_02'] + C * consolidated['pred_03'])
    
    
    # we add A, B, C to the div_by columns if the prediction for corresponding model is not at 0
    consolidated['div_by'] = np.nan
    consolidated['div_A'] = np.round(np.where(consolidated['pred_01'] > 0, A, 0), 1)
    consolidated['div_B'] = np.round(np.where(consolidated['pred_02'] > 0, B, 0), 1)
    consolidated['div_C'] = np.round(np.where(consolidated['pred_03'] > 0, C, 0), 1)
    consolidated[['div_by']] = consolidated['div_A'] + consolidated['div_B'] + consolidated['div_C']
    
    # we do the total with the coefficient divided by the good number according to coefficient and non null values
    consolidated['consolidated_estimation'] = consolidated['total'] / consolidated['div_by']
    
    # we drop unused columns
    consolidated.drop(['div_A','div_B','div_C','div_by', 'total'],axis=1, inplace=True)
    
    return consolidated


def meatrics_mse_rmse_consolidated(df):
    'Function used to return meam square error and root mean square error'
    'for a dataframe for which we know the real value (real_rate) and the predicted one (consolidated_estimation)'
    
    # Real values are in the column called : 'real_rank_from_......'
    realVals = df['real_rank']

    # Predicted values are in the column called :  'consolidated_estimation'
    predictedVals = df['consolidated_estimation']

    #calucul of mean squared error
    mse = mean_squared_error(realVals, predictedVals)
    
    #calucul of root mean squared error
    rmse = mean_squared_error(realVals, predictedVals, squared = False)
    
    return mse, rmse


###########
###########
###########
## END of Function_05
###########
###########
###########