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
import correlation_stats
import alternative_risk_premium

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



def update_benchmark_index():
    df_asset_index = pd.read_sql("SELECT * FROM AssetIndex",acc_engine).sort_values("ID_INSTRUMENT")
    df_asset_index = df_asset_index[df_asset_index["Instrument_Type"] != "Index"]
    df_benchmark_index = pd.read_sql("SELECT * FROM BenchmarkIndex",acc_engine).sort_values("ID_INSTRUMENT")
    asset_ids = list(df_asset_index["ID_INSTRUMENT"].unique())

    def update_market():
        df_market_bench = df_benchmark_index[df_benchmark_index["Industry"] == "Market"]
        bench_ids = list(df_market_bench["ID_INSTRUMENT"].unique())
        missing_ids = [x for x in asset_ids if x not in bench_ids]

        for i in missing_ids:
            value_list = [10,"US_SP500",i,"US_Market","Market"]
            df_bench_ind = pd.DataFrame([value_list], columns = df_benchmark_index.columns)
            df_bench_ind.set_index("ID_BENCHMARK",inplace = True)
            df_bench_ind.to_sql("BenchmarkIndex",acc_engine,if_exists = "append")
            
        pass

    def update_industry():
        pass
    update_market()

#update_benchmark_index()

def update_correlations(asset_index):
    industry_unique = list(asset_index["Industry"].unique())
    for x in industry_unique:
        asset_index_industry = asset_index[asset_index["Industry"] == x]
        ids = list(asset_index_industry["ID_INSTRUMENT"].values)
        print(ids)
        if len(ids) >=2:
            for i in range(len(ids)):
                pair_1 = ids[i]
                ids_plus_1 = ids[i:len(ids)]
                for z in ids_plus_1:
                    if pair_1 == z:
                        pass
                    else:
                        correlation_stats.pair_correlation(pair_1,z,1)
        else:
            pass


def update_betas(asset_index):
    benchmark_index = pd.read_sql("SELECT * FROM BenchmarkIndex",acc_engine)
    unique_benchmarks = benchmark_index["ID_BENCHMARK"].unique()

    for i in unique_benchmarks:
        benchmark_index_ind = benchmark_index[benchmark_index["ID_BENCHMARK"] ==i]
        for x in range(len(benchmark_index_ind)):
            id_instrument = benchmark_index_ind[benchmark_index_ind["ID_INSTRUMENT"] ==benchmark_index_ind.iloc[x]["ID_INSTRUMENT"]]["ID_INSTRUMENT"].values[0]
            correlation_stats.beta_estimation(id_instrument,i,"D",1)



    #industry_unique = list(asset_index["Industry"].unique())

def update_cointegration_check(asset_index):
    industry_unique = list(asset_index["Industry"].unique())
    industry_unique.remove("Cryptocurrency")
    #industry_unique = ["Restaurants","REIT—Healthcare Facilities","Telecom and Entertainment","Airlines","Oil & Gas Integrated","Software and Information"]
    for x in industry_unique:
        asset_index_industry = asset_index[asset_index["Industry"] == x]
        ids = list(asset_index_industry["ID_INSTRUMENT"].values)
        if len(ids) >=2:
            for i in range(len(ids)):
                pair_1 = ids[i]
                ids_plus_1 = ids[i:len(ids)]
                for z in ids_plus_1:
                    if pair_1 == z:
                        pass
                    else:
                        print(pair_1)
                        print(z)
                        print(x)
                        pair_coint =correlation_stats.cointegration_analysis(pair_1,z,x,1,"D","expanding")
                        if pair_coint == "True":
                            correlation_stats.rolling_cointegration(pair_1,z,x,"D",1)
                        else:
                            pass
        else:
            pass


def main_update():
    index_asset = pd.read_sql("SELECT * FROM AssetIndex",acc_engine).sort_values("ID_INSTRUMENT")
    update_cointegration_check(index_asset)
    update_betas(index_asset)
    update_correlations(index_asset)
    alternative_risk_premium.momentum_premium_rank("D",1)

#main_update() # 1256 seconds full UPDATE



def label_momentum():
    alternative_risk_premium.momentum_premium_rank("D",1)


def label_beta():
    df_betas = pd.read_sql("SELECT * FROM BetaIndex",acc_engine).sort_values("ID_INSTRUMENT")
    unique_ids = list(df_betas["ID_INSTRUMENT"].unique())
    unique_betas = list(df_betas["Beta_Type"].unique())

    for x in unique_betas:
        df_beta_rank_exists = pd.read_sql("SELECT * FROM RankTable WHERE Rank_Type = '{}' AND Benchmark = '{}'".format("Beta",x),acc_engine)
        unique_rank_ids = list(df_beta_rank_exists["ID_INSTRUMENT"].unique())
        

        ## Pegamos nos ids dos novos ids adicionados dos betas e os ids dos rankings existentes 
        ## QUando são diferentes, apagamos os ranks por inteiro e recorremos tudo

        if len(unique_ids) != len(unique_rank_ids):
            delete_table_query = """
                DELETE FROM RankTable 
                WHERE Benchmark in ('{}')
            """.format(x)

            conn_str_2 = pyodbc.connect(conn_str)
            execute_query(conn_str_2,delete_table_query)
        else:
            print("No new IDs were added")
            pass
        df_beta_type = df_betas[df_betas["Beta_Type"] ==x]
        df_beta_type = df_beta_type.sort_values("Data")
        time.sleep(2)

        ## VOltamos a chamar a  base de dados para verificar se eles forama apgados ou náo. se sim corremos tudo, para este beta, se não, prosseguimos
        df_beta_rank_exists = pd.read_sql("SELECT * FROM RankTable WHERE Rank_Type = '{}' AND Benchmark = '{}'".format("Beta",x),acc_engine)
        if df_beta_rank_exists.empty:
            df_beta_type["Rank_Value"] =df_beta_type.groupby('Data')['Beta'].rank(ascending=True)
            df_beta_rank = df_beta_type[["ID_INSTRUMENT","Data","Rank_Value","Beta_Type"]].sort_values("Data")
            df_beta_rank["Rank_Type"] = "Beta"
            df_beta_rank["Benchmark"] = x
            df_beta_rank.drop(columns = "Beta_Type",inplace = True)
            df_beta_rank.set_index("ID_INSTRUMENT",inplace = True)
            df_beta_rank.to_sql("RankTable",acc_engine,if_exists = "append")

        else:
            last_date = df_beta_rank_exists.iloc[-1]["Data"]
            df_beta_type = df_beta_type[df_beta_type["Data"]>last_date]
            df_beta_type["Rank_Value"] =df_beta_type.groupby('Data')['Beta'].rank(ascending=True)
            df_beta_rank = df_beta_type[["ID_INSTRUMENT","Data","Rank_Value","Beta_Type"]].sort_values("Data")
            df_beta_rank["Rank_Type"] = "Beta"
            df_beta_rank["Benchmark"] = x
            df_beta_rank.drop(columns = "Beta_Type",inplace = True)
            df_beta_rank.set_index("ID_INSTRUMENT",inplace = True)
            df_beta_rank.to_sql("RankTable",acc_engine,if_exists = "append")

        ## Continue with the renaming for the append of the data.
        # This time, is easier for the rank append overtime.

#label_beta()




def main_update_ranks():
    ## New ranks needs to be created, everytime:
    ## A new Stock is Added or REmoved
    #label_price_stats()
    main_update()
    label_beta()

main_update_ranks()