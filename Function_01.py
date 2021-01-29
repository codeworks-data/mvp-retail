import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import datetime

###########
###########
#function for all notebook 
###########
###########

def hist(datas, col, rot=False):
    'Function to display histogram with a df and a column name'
    'We can put true or false if we want to rotate x labels'
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


def enhance_product_df(df):
    'This function add two columns to the data frame mapping full object_id and full object_name'
    df['prod_cat_code'] = df['prod_cat_code'].astype(str)
    df['prod_sub_cat_code'] = df['prod_sub_cat_code'].astype(str)
    df['object_id'] = df['prod_cat_code'] + '_' + df['prod_sub_cat_code']
    df['object_name'] = df['prod_cat'] + '_' + df['prod_subcat']
    return df

def create_dict_object(df, object_id, object_name):
    'Function to create a dictionnay between object_id and object_name'
    return dict(zip(df.object_id,df.object_name))

###########
###########
## functions for part III. Focus on customer information
###########
###########

def hist(datas, col, rot=False):
    'Function to display histogram with a df and a column name'
    'We can put true or false if we want to rotate x labels'
    sns.countplot(x=col, data=datas)
    if rot == True:
        plt.xticks(rotation=90)
    else:
        pass
    plt.show()
    
def age(val):
    'Function to obtain the age of the person from his Date of Birth written in a str'
    day = int(val[:2])
    mounth = int(val[3:5]) 
    year = int(val[6:10])
    
    t_start = datetime.datetime(year,mounth,day)    
    t_end = datetime.datetime.now()
    return int((t_end - t_start).total_seconds()/(3600*24*30*12))  



def function_parse_date(a):
    'This function is used in order to have all date with the same format which is dd-mm-yyyy'
    L = []
    n = 0
    for i in a:
        
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


def age_from_transaction(df):
    'Function which add a columns in the df with the age of the customer when he buy the object'
    df_T = pd.read_csv("Transactions.csv")
    df_T['tran_date'] = df_T['tran_date'].apply(function_parse_date)
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


def modification_on_transcation(df, dico):
    'function to run some augmentation on transction data frame'
    df.drop(['transaction_id', 'Rate', 'Tax', 'total_amt'], axis=1, inplace=True)
    df['prod_cat_code'] = df['prod_cat_code'].astype(str)
    df['prod_subcat_code'] = df['prod_subcat_code'].astype(str)
    df['object_id'] = df['prod_cat_code'] + '_' + df['prod_subcat_code']
    df['object_name'] = df['object_id'].map(dico) 
    df.rename({'Qty': 'rank'}, axis=1, inplace = True)
    df = df[df['rank']>0]
    df = df.reset_index(drop = True)
    return df


def barplot_display_cat(data, ab, ordo):
    'Function to draw a barplot'
    plt.figure(figsize=(10,6))
    sns.barplot(x=data[ab], y=data[ordo], palette="Reds_r")
    plt.xlabel('\n Category and SubCat', fontsize=15, color='#c0392b')
    plt.ylabel(" Mean rank by item\n", fontsize=15, color='#c0392b')
    plt.xticks(rotation= 90)
    plt.tight_layout()
    


###########
###########
##functions for the part V. Creation of the item-user matrix 
###########
###########


def get_rank_count_logcount(df):
    'function to add a column with the log count'
    
    num_users = len(df.cust_id.unique())
    num_items = len(df.object_id.unique())

    # there are a lot more counts in rating of zero
    total_cnt = num_users * num_items
    rating_zero_cnt = total_cnt - df.shape[0]

    # get count
    df_ratings_cnt_tmp = pd.DataFrame(df.groupby('rank').size(), columns=['count'])

    df_ratings_cnt = df_ratings_cnt_tmp.append(
        pd.DataFrame({'count': rating_zero_cnt}, index=[0.0]),
        verify_integrity=True,
    ).sort_index()

    #log normalise to make it easier to interpret on a graph
    
    df_ratings_cnt['log_count'] = np.log(df_ratings_cnt['count'])
    return df_ratings_cnt



    
def barplot_display_rank(data, ab, ordo, log):
    'function to display rank count and log rank count'
    plt.figure(figsize=(10,6))
    sns.barplot(x=data[ab], y=data[ordo], palette="Reds_r")
    plt.xlabel('\n Rank', fontsize=15, color='#c0392b')
    if log == True:
        plt.ylabel("How many rank ? (in log) \n", fontsize=15, color='#c0392b')
    else:
        plt.ylabel("How many rank ? \n", fontsize=15, color='#c0392b')
    plt.xticks(rotation= 45)
    plt.tight_layout()






