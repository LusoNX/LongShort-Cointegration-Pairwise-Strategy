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
import json
import correlation_stats

import database_directory
conn_str = database_directory.main()
cnn_url = f"access+pyodbc:///?odbc_connect={urllib.parse.quote_plus(conn_str)}"
acc_engine = create_engine(cnn_url)


## Check the highest correlated instruments, within an industry.
def correlation_filter(df_correlations):
    unique_pairs = list(df_correlations["Pair"].unique())
    dict_pairs_corr = {}
    for x in unique_pairs:
        df_correlation_ind = df_correlations[df_correlations["Pair"] == x]
        dict_pairs_corr[x] =df_correlation_ind["Correlation"].describe()

    # This 1st specifies the min requirement for the assets to be in the strategy
    df_corr_desc = pd.DataFrame(dict_pairs_corr)
    def filter_1(DF,DF_DESC,min_correl_recent,min_correl_p25,min_correl_p50,max_std):
        df = DF.copy()
        last_date = df.iloc[-1]["Data"]
        df_filter = df[df["Data"] == last_date]
        df_filter = df_filter[df_filter["Correlation"] >=min_correl_recent]
        symbols_filtered = list(df_filter["Pair"].unique())
        df_desc = DF_DESC.copy()
        df_desc =df_desc[symbols_filtered].T
        df_filter_final = df_desc[df_desc["25%"] >= min_correl_p25]
        df_filter_final = df_filter_final[df_filter_final["50%"] >= min_correl_p50]
        df_filter_final = df_filter_final[df_filter_final["std"] <= max_std]
        return list(df_filter_final.index.values)

    symbol_filtered = filter_1(df_correlations,df_corr_desc,0.8,0.6,0.75,0.07)
    return symbol_filtered







def beta_neutrality(id_1,id_2,beta_type):
    # HERE, the goal is to test in-smaple betas, and check whether is possibible to derive some sort of information, relating their predictability.
    def get_short_long(row):
        if row >=0:
            value = -1
        else:
            value = 1
        return value
    ## First for the overall beta_market
    df_betas = pd.read_sql("SELECT * FROM BetaIndex WHERE Beta_Type ='{}'".format(beta_type),acc_engine).sort_values("Data")
    df_beta_1 = df_betas[df_betas["ID_INSTRUMENT"] == id_1][["Data","Beta"]]#.set_index("Data",inplace = True)
    df_beta_2 = df_betas[df_betas["ID_INSTRUMENT"] == id_2][["Data","Beta"]]#.set_index("Data",inplace = True)
    df_beta_1.set_index("Data",inplace = True)
    df_beta_2.set_index("Data",inplace = True)
    df_merge = df_beta_1.merge(df_beta_2,how = "inner",right_index = True, left_index = True)
    df_merge = df_merge[~df_merge.index.duplicated(keep='first')]

    # When difference of betas is positive, you are long on the stock 1 from paira and short on the stock2. The opposite is true when diff in betas is negative.
    df_merge["diff_beta"] = df_merge["Beta_x"] -df_merge["Beta_y"]
    df_merge["weight_1"] = df_merge["Beta_y"].abs() /(df_merge["Beta_x"].abs() +df_merge["Beta_y"].abs())
    df_merge["weight_2"] = 1-df_merge["weight_1"]
    df_merge["long_or_short"] = df_merge["diff_beta"].apply(get_short_long)
    
    return df_merge
    

#beta_neutrality(1,5,"US_SP500")

