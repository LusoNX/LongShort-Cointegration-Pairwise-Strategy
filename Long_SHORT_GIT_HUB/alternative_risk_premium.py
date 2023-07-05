from scipy.optimize import minimize
import numpy as np
import pandas as pd
import datetime as dt
import copy
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import sklearn
import time
import math
import plotly.express as px
import plotly.graph_objects as go
from cvxpy import *
from scipy.optimize import nnls
import scipy.optimize
from datetime import datetime
from scipy.optimize import nnls
import seaborn as sns
import urllib
from sqlalchemy import create_engine
import pyodbc
import yfinance as yf
import investpy
from statsmodels.tsa.stattools import adfuller

import database_directory
conn_str = database_directory.main()
cnn_url = f"access+pyodbc:///?odbc_connect={urllib.parse.quote_plus(conn_str)}"
acc_engine = create_engine(cnn_url)


def execute_query(connection, query):
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        connection.commit()
        print("Query successful")
    except OSError as err:
        print(f"Error: '{err}'")

def momentum_inputs(industry,periodicity,rolling_window_years):
    ## Here, i use the Residual Momentum, which better captures the true overperformance of a given stock.
    ## The way this is done is by filtering for the Beta at the industry and size levels, so that the final residual, truly reflects the best performance, after ajusting for systematic risk

    # 1st we get the industry index and their stocks
    asset_index =pd.read_sql("SELECT * FROM AssetIndex",acc_engine)
    benchmark_index = pd.read_sql("SELECT * FROM BenchmarkIndex WHERE Industry = '{}'".format(industry),acc_engine)
    stock_ids = list(benchmark_index["ID_INSTRUMENT"].unique())
    industry_id = benchmark_index["ID_BENCHMARK"].unique()[0]
    region = asset_index[asset_index["ID_INSTRUMENT"] == industry_id]["Region"].values[0]

    ## 2nd we get the small and large cap IDS
    small_cap_id = asset_index[(asset_index["Region"] == region) & (asset_index["Industry"] == "Small Cap")]["ID_INSTRUMENT"].values[0]
    large_cap_id = asset_index[(asset_index["Region"] == region) & (asset_index["Industry"] == "Large Cap")]["ID_INSTRUMENT"].values[0]

    ## Finally, compute the returns for the alfas
    df_prices = pd.read_sql("SELECT * FROM PriceIndex",acc_engine,index_col = "Data").sort_index()
    df_industry = df_prices[df_prices["ID_INSTRUMENT"] == industry_id][["Adj_Close"]]
    df_small_cap = df_prices[df_prices["ID_INSTRUMENT"] == small_cap_id][["Adj_Close"]]
    df_large_cap =df_prices[df_prices["ID_INSTRUMENT"] == large_cap_id][["Adj_Close"]]
    df_industry.rename(columns ={"Adj_Close":"industry"} ,inplace = True)
    df_small_cap.rename(columns ={"Adj_Close":"small_cap"} ,inplace = True)
    df_large_cap.rename(columns ={"Adj_Close":"large_cap"} ,inplace = True)
    df_merge =  df_industry.merge(df_small_cap,how = "inner",right_index = True,left_index = True)
    df_merge =  df_merge.merge(df_large_cap,how = "inner",right_index = True,left_index = True)
    df_merge = df_merge.sort_index()

    if periodicity == "D":
        rolling_window = rolling_window_years*250
        period_v = 250
    elif periodicity == "W":
        df_merge = df_merge.resample(periodicity).last()
        rolling_window = rolling_window_years*52
        period_v = 52
    else:
        pass

    for i in stock_ids:
        df_stock = df_industry = df_prices[df_prices["ID_INSTRUMENT"] == i][["Adj_Close"]]
        df_stock = df_stock.resample(periodicity).last()
        df_merge_n = df_merge.merge(df_stock,how = "inner",right_index = True,left_index = True)
        df_merge_n = df_merge_n.pct_change()
        df_merge_n = df_merge_n.iloc[1::]
        df_merge_n["size_premium"] = df_merge_n["small_cap"] - df_merge_n["large_cap"]
        df_list = []
        df_beta_exists = pd.read_sql("SELECT * FROM IndustryAlfaBeta WHERE ID_INSTRUMENT = {}".format(i),acc_engine).sort_values("Data")
        print(i)
        if df_beta_exists.empty:
            index_values = df_merge_n.iloc[-(len(df_merge_n)-rolling_window+1)::].index.values

            for x in range(len(df_merge_n)-rolling_window+1):
                df_merge_ind =df_merge_n.iloc[0+x:rolling_window+x]
                model = LinearRegression(fit_intercept = True)
                X = np.array(df_merge_ind[["industry","size_premium"]]).reshape(-1,2)
                Y = np.array(df_merge_ind["Adj_Close"]).reshape(-1,1)
                model_fit = model.fit(X,Y)
                alfa = model_fit.intercept_[0]
                beta_industry = model_fit.coef_[0][0]
                beta_size = model_fit.coef_[0][1]
                r2 = model_fit.score(X,Y)
                list_of_values = [alfa,beta_industry,beta_size,r2]
                df_list.append(list_of_values)


            df_beta_final = pd.DataFrame(df_list,columns = ["Alfa","Beta_Industry","Beta_Size","R2"])
            df_beta_final["Data"] = index_values
            df_beta_final["ID_INSTRUMENT"] = i
            df_beta_final.set_index("ID_INSTRUMENT",inplace = True)
            df_beta_final.to_sql("IndustryAlfaBeta",acc_engine,if_exists = "append")
            ## ALso append to the other dataframe 

            
        else:
            last_beta_frame = df_beta_exists["Data"].iloc[-1]
            new_range_len = len(df_merge_n[df_merge_n.index > last_beta_frame])
            beta_index = df_merge_n[df_merge_n.index > last_beta_frame].index.values
            df_merge_new = df_merge_n.iloc[-(new_range_len+period_v)+1::]

            for x in range(new_range_len):
                df_merge_new_ind = df_merge_new.iloc[0+i:period_v+i]
                model = LinearRegression(fit_intercept = True)
                X = np.array(df_merge_new_ind[["industry","size_premium"]]).reshape(-1,2)
                Y = np.array(df_merge_new_ind["Adj_Close"]).reshape(-1,1)
                model_fit = model.fit(X,Y)
                alfa = model_fit.intercept_[0]
                beta_industry = model_fit.coef_[0][0]
                beta_size = model_fit.coef_[0][1]
                r2 = model_fit.score(X,Y)
                list_of_values = [alfa,beta_industry,beta_size,r2]
                df_list.append(list_of_values)

            df_beta_final = pd.DataFrame(df_list,columns = ["Alfa","Beta_Industry","Beta_Size","R2"])
            df_beta_final["Data"] = beta_index
            df_beta_final["ID_INSTRUMENT"] = i
            df_beta_final.set_index("ID_INSTRUMENT",inplace = True)
            df_beta_final.to_sql("IndustryAlfaBeta",acc_engine,if_exists = "append")
        
            pass


def momentum_inputs_update(periodicity):
    list_of_industries = list(pd.read_sql("SELECT Industry FROM BenchmarkIndex",acc_engine)["Industry"].unique())
    list_of_industries.remove("Market")
    list_of_industries.remove("US_Market")
    for i in list_of_industries:
        momentum_inputs(i,periodicity,1)

#momentum_inputs_update("D")

def momentum_premium_rank(periodicity,momentum_years):
    ## This is done via the residual momentum, so its possible to make a different classification to identify winners from losers within industry
    ## In this case, we choose to create a classification for all stocks under analysis. Not by industry. It should identify winners from losers, regardless of their industry
    momentum_inputs_update("D")
    df_alfas = pd.read_sql("SELECT * FROM IndustryAlfaBeta",acc_engine).sort_values("ID_INSTRUMENT")
    unique_ids = list(df_alfas["ID_INSTRUMENT"].unique())
    beta_index = pd.read_sql("SELECT * FROM BenchmarkIndex",acc_engine)
    beta_index = beta_index[beta_index["Industry"]!= "Market"]
    df_alfas = df_alfas.sort_values("Data")

    df_momentum_exists = pd.read_sql("SELECT * FROM RankTable WHERE Rank_Type = '{}'".format("Momentum"),acc_engine).sort_values("ID_INSTRUMENT")
    momentum_rank_ids = list(df_momentum_exists["ID_INSTRUMENT"].unique())
   
    if periodicity == "D":
        rolling_window = momentum_years*250
        #period_v = 250
    elif periodicity == "W":
        df_merge = df_merge.resample(periodicity).last()
        rolling_window = momentum_years*52
        #period_v = 52
    else:
        pass
    df_alfas.set_index("Data",inplace = True)
    df_alfas = df_alfas.sort_index()
    unique_data_index = list(df_alfas.index.unique())
    df_momentum = pd.DataFrame(index = unique_data_index)
    df_momentum["Data"] = unique_data_index

    if unique_ids != momentum_rank_ids:
        delete_table_query = """
            DELETE FROM RankTable 
            WHERE Rank_Type in ('{}')
        """.format("Momentum")

        conn_str_2 = pyodbc.connect(conn_str)
        execute_query(conn_str_2,delete_table_query)
    else:
        print("No new IDs were added")
        pass
   
    df_momentum_exists = pd.read_sql("SELECT * FROM RankTable WHERE Rank_Type = '{}'".format("Momentum"),acc_engine).sort_values("Data")

    if df_momentum_exists.empty:
        for i in unique_ids:
            df_alfa_v = df_alfas[df_alfas["ID_INSTRUMENT"] == i]
            range_v = len(df_alfa_v)- rolling_window
            alfa_v_list = []
            new_index = df_alfa_v.iloc[-(range_v+1)::].index.values
            for x in range(range_v+1):
                df_alfa_ind = df_alfa_v.iloc[0+x:rolling_window+x+1]
                alfa_v= (1+df_alfa_ind["Alfa"]).cumprod()[-1]-1
                alfa_v_list.append(alfa_v)
            df_anualized_alfa = pd.DataFrame(alfa_v_list, columns = [i])
            df_anualized_alfa["Data"] = new_index
            df_momentum = df_momentum.merge(df_anualized_alfa[pd.notnull(df_anualized_alfa.Data)], how='left', on="Data")
            df_momentum = df_momentum.sort_values("Data")
            df_momentum.set_index("Data",inplace = True)

            
    else:
        last_date = df_momentum_exists.iloc[-1]["Data"]
           
        df_momentum_n = df_momentum[df_momentum.index >last_date]
     

        for i in unique_ids:
            df_alfa_v = df_alfas[df_alfas["ID_INSTRUMENT"] == i]
            range_n = df_alfa_v[df_alfa_v.index>last_date]
            new_range = len(range_n)+rolling_window
            df_alfa_v = df_alfa_v.iloc[-new_range::]
            range_v = len(range_n)

            alfa_v_list = []
            new_index = df_alfa_v.iloc[-(range_v)::].index.values
            
            for x in range(range_v):
                df_alfa_ind = df_alfa_v.iloc[0+x:rolling_window+x+1]
                alfa_v= (1+df_alfa_ind["Alfa"]).cumprod()[-1]-1
                alfa_v_list.append(alfa_v)
            
            df_anualized_alfa = pd.DataFrame(alfa_v_list, columns = [i])
            df_anualized_alfa["Data"] = new_index
            df_momentum = df_momentum.merge(df_anualized_alfa[pd.notnull(df_anualized_alfa.Data)], how='left', on="Data")
            df_momentum.sort_values("Data",inplace = True)
            df_momentum.set_index("Data",inplace = True)
        df_momentum = df_momentum[df_momentum.index >last_date]
       
    if df_momentum.empty:
        pass
    else:

        df_momentum_rank = df_momentum.rank(axis =1, ascending = False)
        
        for i in range(len(df_momentum_rank.columns)):
            id_instrument = df_momentum_rank.columns[i]
            df_ind = df_momentum_rank[[df_momentum_rank.columns[i]]]
            df_ind["ID_INSTRUMENT"] = id_instrument
            df_ind["Rank_Type"] = "Momentum"
            df_ind.rename(columns = {df_momentum_rank.columns[i]:"Rank_Value"},inplace = True)
            df_ind["Benchmark"] = beta_index[beta_index["ID_INSTRUMENT"] ==id_instrument]["Benchmark_Name"].values[0]
            df_ind.reset_index(inplace  = True)
            df_ind.set_index("ID_INSTRUMENT",inplace = True)
            df_ind.dropna(inplace = True)
            df_ind.to_sql("RankTable",acc_engine,if_exists = "append")

        return df_momentum_rank
    
#momentum_premium_rank("D",1)

