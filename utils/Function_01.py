import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import datetime

###########
###########
## function for all notebook 
###########
###########

def hist(datas, col, rot=False):
    'Function to display a seaborn countplot from a df and column name'
    'We can set rot to true or false if we want to rotate x labels'
    sns.countplot(x=col, data=datas)
    if rot == True:
        plt.xticks(rotation=90)
    else:
        pass
    plt.show()
    

###########
###########
## function for part II. Focus on product
###########
###########


def adding_name_ID_to_product_df(df):
    'Function which add two new columns to the df which are the full object_id and the full object_name'
    df['prod_cat_code'] = df['prod_cat_code'].astype(str)
    df['prod_sub_cat_code'] = df['prod_sub_cat_code'].astype(str)
    df['object_id'] = df['prod_cat_code'] + '_' + df['prod_sub_cat_code']
    df['object_name'] = df['prod_cat'] + '_' + df['prod_subcat']
    return df

def create_dict_object_ID_Name(df, object_id, object_name):
    'Function to create a dictionnay between object_id and object_name'
    return dict(zip(df.object_id,df.object_name))

###########
###########
## functions for part III. Focus on customer information
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


def age_when_purchase(df):
    'Function which add a columns in the df with the age of the customer when he bought the object'
    df_T = pd.read_csv("data/Transactions.csv")
    df_T['tran_date'] = df_T['tran_date'].apply(reformat_date)
    df = pd.merge(left = df_T,
                         right = df,
                         left_on = "cust_id",
                         right_on = "customer_Id",
                         how = "left")
    df['DOB'] = pd.to_datetime(df.DOB, format='%d-%m-%Y') 
    df['tran_date'] = pd.to_datetime(df.tran_date, format='%d-%m-%Y')
    df['diff_trans__DOB'] = (df.tran_date-df.DOB)
    df['age_when_purchase'] = df['diff_trans__DOB'].map(lambda x:x.days)//360.25
    
    return df[['customer_Id', 'Gender', 'DOB', 'age_when_purchase', 'city_code']]


###########
###########
## functions for part IV. Focus on transaction
###########
###########


def transaction_final_df(df, dico):
    'Function to run some augmentation on transaction data frame'
    
    #we drop unused columns
    df.drop(['transaction_id', 'Rate', 'Tax', 'total_amt'], axis=1, inplace=True)
    
    #we change type to str
    df['prod_cat_code'] = df['prod_cat_code'].astype(str)
    df['prod_subcat_code'] = df['prod_subcat_code'].astype(str)
    
    #we add new columns with object id and object name
    df['object_id'] = df['prod_cat_code'] + '_' + df['prod_subcat_code']
    df['object_name'] = df['object_id'].map(dico) 
    
    #we rename Qty with rating
    df.rename({'Qty': 'rating'}, axis=1, inplace = True)
    
    #we only keep rating above 0
    df = df[df['rating']>0]
    df = df.reset_index(drop = True)
    
    
    return df


def barplot_mean_rate_by_item(data, ab, ordo):
    'Function to draw a barplot according to two columns'
    plt.figure(figsize=(10,6))
    sns.barplot(x=data[ab], y=data[ordo], palette="Reds")
    plt.xlabel('\n Category and SubCat', fontsize=15, color='#c0392b')
    plt.ylabel(" Mean rank by item\n", fontsize=15, color='#c0392b')
    plt.xticks(rotation= 90)
    plt.tight_layout()
    


###########
###########
##functions for the part V. Creation of the item-user matrix 
###########
###########


def get_rate_logcount(df):
    'Function to add a column with the log count'
    
    num_users = len(df.cust_id.unique())
    num_items = len(df.object_id.unique())

    # there are a lot more counts in rating of zero
    total_cnt = num_users * num_items
    rating_zero_cnt = total_cnt - df.shape[0]

    # get count
    df_ratings_cnt_tmp = pd.DataFrame(df.groupby('rating').size(), columns=['count'])

    df_ratings_cnt = df_ratings_cnt_tmp.append(
        pd.DataFrame({'count': rating_zero_cnt}, index=[0.0]),
        verify_integrity=True,
    ).sort_index()

    #log normalise to make it easier to interpret on a graph
    
    df_ratings_cnt['log_count'] = np.log(df_ratings_cnt['count'])
    return df_ratings_cnt



    
def barplot_rate_count(data, ab, ordo, log):
    'Function to display rate count and log of the rate count'
    plt.figure(figsize=(10,6))
    sns.barplot(x=data[ab], y=data[ordo], palette="Reds_r")
    plt.xlabel('\n Ratings', fontsize=15, color='#c0392b')
    if log == True:
        plt.ylabel("How many rate (in log) \n", fontsize=15, color='#c0392b')
    else:
        plt.ylabel("How many rate \n", fontsize=15, color='#c0392b')
    plt.xticks(rotation= 45)
    plt.tight_layout()


###########
###########
###########
## END of Function_01
###########
###########
###########
