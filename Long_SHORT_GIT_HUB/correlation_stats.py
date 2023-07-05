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

def pair_correlation(id_1,id_2,rolling_window):
    asset_index = pd.read_sql("SELECT ID_INSTRUMENT,Symbol,Industry FROM AssetIndex",acc_engine)
    industry_name = asset_index[asset_index["ID_INSTRUMENT"]== id_1]["Industry"].values[0]
    symbol_1 =asset_index[asset_index["ID_INSTRUMENT"]== id_1]["Symbol"].values[0]
    symbol_2 =asset_index[asset_index["ID_INSTRUMENT"]== id_2]["Symbol"].values[0]
    pair_name = symbol_1+"/"+symbol_2
    df_price_1 = pd.read_sql("SELECT ID_INSTRUMENT,Data,Adj_Close FROM PriceIndex WHERE ID_INSTRUMENT = {}".format(id_1),acc_engine).sort_values("Data")
    df_price_2 = pd.read_sql("SELECT ID_INSTRUMENT,Data,Adj_Close FROM PriceIndex WHERE ID_INSTRUMENT = {}".format(id_2),acc_engine).sort_values("Data")
    df_price_1["returns_1"] = df_price_1["Adj_Close"].pct_change()
    df_price_2["returns_2"] = df_price_2["Adj_Close"].pct_change()
    rolling_window_v = rolling_window*250
    df_price_1.set_index("Data",inplace = True)
    df_price_2.set_index("Data",inplace = True)
    df_price_1 = df_price_1[["returns_1"]]
    df_price_2 = df_price_2[["returns_2"]]

    df_merge = df_price_1.merge(df_price_2,how = "inner",right_index = True,left_index = True)
    corr_index = df_price_1.iloc[-(len(df_merge)-rolling_window_v+1)::].index.values
    corr_list = []
    df_correlations_exist = pd.read_sql("SELECT * FROM CorrelationIndex WHERE Pair ='{}' ".format(pair_name),acc_engine).sort_values("Data")

    if df_correlations_exist.empty:
        for i in range(len(df_merge)-rolling_window_v+1):
            df_merge_ind = df_merge.iloc[0+i:rolling_window_v+i]
            corr_values = df_merge_ind["returns_1"].corr(df_merge_ind["returns_2"])
            corr_list.append(corr_values)
        df_correlations = pd.DataFrame(corr_list,index =corr_index,columns = ["Correlation"])
        df_correlations["Pair"] = pair_name
        df_correlations.index.name = "Data"
        df_correlations.sort_index(inplace = True)
        df_correlations["Industry"] = industry_name
        df_correlations.to_sql("CorrelationIndex",acc_engine,if_exists = "append")
    else:
        last_date = df_correlations_exist.iloc[-1]["Data"]
        new_range_len = len(df_merge[df_merge.index > last_date])
        corr_index = df_merge[df_merge.index > last_date].index.values
        df_merge_new = df_merge.iloc[-(new_range_len+250)::]
        for i in range(new_range_len):
            df_merge_new_ind = df_merge_new.iloc[0+i:250+i]
            corr_values = df_merge_new_ind["returns_1"].corr(df_merge_new_ind["returns_2"])
            corr_list.append(corr_values)
        df_correlations = pd.DataFrame(corr_list,index =corr_index,columns = ["Correlation"])
        df_correlations["Pair"] = pair_name
        df_correlations.index.name = "Data"
        df_correlations.sort_index(inplace = True)
        df_correlations["Industry"] = industry_name

        df_correlations.to_sql("CorrelationIndex",acc_engine,if_exists = "append")

    print("Correlation Data for ID: {}/{}  was updated".format(id_1,id_2))
    
#pair_correlation(1,2,1)

def beta_estimation(id_instrument,id_market,periodicity,rolling_window_years):
    df_price_1 = pd.read_sql("SELECT ID_INSTRUMENT,Data,Adj_Close FROM PriceIndex WHERE ID_INSTRUMENT = {}".format(id_instrument),acc_engine).sort_values("Data")
    df_market = pd.read_sql("SELECT ID_INSTRUMENT,Data,Adj_Close FROM PriceIndex WHERE ID_INSTRUMENT = {}".format(id_market),acc_engine).sort_values("Data")

    if periodicity == "D":
        rolling_window = rolling_window_years*250
        period_v = 250
    elif periodicity == "W":
        df_price_1 = df_price_1.resample(periodicity).last()
        df_price_2 = df_price_2.resample(periodicity).last()
        rolling_window = rolling_window_years*52
        period_v = 52
    else:
        pass

    BETA_TYPE = pd.read_sql("SELECT ID_INSTRUMENT,Benchmark_Name FROM BenchmarkIndex WHERE ID_BENCHMARK = {}".format(id_market),acc_engine)["Benchmark_Name"].values[0]
    df_price_1["returns_1"] = df_price_1["Adj_Close"].pct_change()
    df_market["returns_bench"] = df_market["Adj_Close"].pct_change()
    rolling_window_v = rolling_window
    df_price_1.set_index("Data",inplace = True)
    df_market.set_index("Data",inplace = True)
    df_price_1 = df_price_1[["returns_1"]]
    df_market = df_market[["returns_bench"]]
    df_merge = df_price_1.merge(df_market,how = "inner",right_index = True,left_index = True)
    df_merge.dropna(inplace = True)
    df_merge = df_merge.sort_index()
    beta_index = df_merge.iloc[-(len(df_merge)-rolling_window_v+1)::].index.values
    beta_list = []
    r2_list = []
    df_beta_exists = pd.read_sql("SELECT * FROM BetaIndex WHERE Beta_Type = '{}' AND ID_INSTRUMENT = {} ".format(BETA_TYPE,id_instrument),acc_engine)
    
    if df_beta_exists.empty:
        for i in range(len(df_merge)-rolling_window_v+1):
            df_merge_ind =df_merge.iloc[0+i:rolling_window_v+i]
            model = LinearRegression(fit_intercept = True)
            Y = np.array(df_merge_ind["returns_1"]).reshape(-1,1)
            X = np.array(df_merge_ind["returns_bench"]).reshape(-1,1)
            model_fit = model.fit(X,Y)
            beta = model_fit.coef_[0][0]
            r2 = model_fit.score(X,Y)
            r2_list.append(r2)
            beta_list.append(beta)

        df_beta_frame = pd.DataFrame([beta_list,r2_list]).T
        df_beta_frame["Data"] = beta_index
        df_beta_frame.rename(columns = {0:"Beta",1:"R2"},inplace = True)
        df_beta_frame["Beta_Type"] = BETA_TYPE
        df_beta_frame["ID_INSTRUMENT"] = id_instrument
        df_beta_frame.set_index("ID_INSTRUMENT",inplace = True)
        df_beta_frame.to_sql("BetaIndex",acc_engine,if_exists = "append")

    else:
        last_beta_frame = df_beta_exists["Data"].iloc[-1]
        new_range_len = len(df_merge[df_merge.index > last_beta_frame])
        beta_index = df_merge[df_merge.index > last_beta_frame].index.values
        df_merge_new = df_merge.iloc[-(new_range_len+period_v)+1::]

        for i in range(new_range_len):
            df_merge_new_ind = df_merge_new.iloc[0+i:period_v+i]
            model = LinearRegression(fit_intercept = True)
            Y = np.array(df_merge_new_ind["returns_1"]).reshape(-1,1)
            X = np.array(df_merge_new_ind["returns_bench"]).reshape(-1,1)
            model_fit = model.fit(X,Y)
            beta = model_fit.coef_[0][0]
            r2 = model_fit.score(X,Y)
            r2_list.append(r2)
            beta_list.append(beta)


        df_beta_frame = pd.DataFrame([beta_list,r2_list]).T
        df_beta_frame["Data"] = beta_index
        df_beta_frame.rename(columns = {0:"Beta",1:"R2"},inplace = True)
        df_beta_frame["Beta_Type"] = BETA_TYPE
        df_beta_frame["ID_INSTRUMENT"] = id_instrument
        df_beta_frame.set_index("ID_INSTRUMENT",inplace = True)
        df_beta_frame.to_sql("BetaIndex",acc_engine,if_exists = "append")

    print("Market Beta Data for ID: {}  was updated".format(id_instrument))
#beta_estimation(1,10,"D",1)

def pairwise_spread(id_1,id_2,minimum_years,periodicity,ma_period):
    df_price_1 = pd.read_sql("SELECT ID_INSTRUMENT,Data,Adj_Close FROM PriceIndex WHERE ID_INSTRUMENT = {}".format(id_1),acc_engine,index_col = "Data").sort_index()
    df_price_2 = pd.read_sql("SELECT ID_INSTRUMENT,Data,Adj_Close FROM PriceIndex WHERE ID_INSTRUMENT = {}".format(id_2),acc_engine,index_col = "Data").sort_index()
    if periodicity == "D":
        rolling_window = minimum_years*250
    elif periodicity == "W":
        df_price_1 = df_price_1.resample(periodicity).last()
        df_price_2 = df_price_2.resample(periodicity).last()
        rolling_window = minimum_years*52
    else:
        pass

    df_price_1["returns_1"] = df_price_1["Adj_Close"].pct_change()
    df_price_2["returns_2"] = df_price_2["Adj_Close"].pct_change()

    df_price_1 = df_price_1[["returns_1"]]
    df_price_2 = df_price_2[["returns_2"]]

    df_merge = df_price_1.merge(df_price_2,how = "inner",right_index = True, left_index = True)
    df_merge["spread"] = df_merge["returns_1"] - df_merge["returns_2"]
    df_merge = df_merge.iloc[1::]
    index_values = df_merge.iloc[-(len(df_merge)-rolling_window+1)::].index.values
    standardized_values = []

    for i in range(len(df_merge)-rolling_window+1):
        df_merge_ind = df_merge.iloc[0:rolling_window+i]
        mean_v = df_merge_ind["spread"].mean()
        std_v = df_merge_ind["spread"].std()
        standardized_v = (df_merge_ind.iloc[-1]["spread"]-mean_v)/std_v
        standardized_values.append(standardized_v)
    df_standardized_values = pd.DataFrame(standardized_values,index = index_values)
    df_standardized_values.rename(columns = {0:"zscore"},inplace = True)
    df_standardized_values["MA_zscore"] = df_standardized_values.zscore.rolling(ma_period, min_periods = 4).mean()
    df_standardized_values.index.name = "Data"
    return df_standardized_values


def multiple_pairwise(id_1,id_2,periodicity_1,periodicity_2,rolling_period_1,rolling_period_2):
    df_standard_d =pairwise_spread(id_1,id_2,1,periodicity_2,rolling_period_2).sort_index()
    df_standard_w =pairwise_spread(id_1,id_2,1,periodicity_1,rolling_period_1).sort_index()
    df_standard_w.rename(columns = {"zscore":"zscore_{}".format(periodicity_1),"MA_zscore":"MA_zscore_{}".format(periodicity_1)},inplace = True)

    df_standard_d.rename(columns = {"zscore":"zscore_{}".format(periodicity_2),"MA_zscore":"MA_zscore_{}".format(periodicity_2)},inplace = True)
    df_standard_w.index = df_standard_w.index -pd.Timedelta(days =2)
    df_standard_d.reset_index(inplace = True)
    df_standard_w.reset_index(inplace = True)
    df_merged = df_standard_d.merge(df_standard_w,on ="Data",how='outer')
    df_merged.set_index("Data",inplace = True)
    df_merged.sort_index(inplace = True)
    
    return df_merged


#multiple_pairwise(1,2,"W","D",4,5)


def adf_test(data_series):
    adf_fuller_results = adfuller(data_series)
    p_value = adf_fuller_results[1]

    if p_value <= 0.05:
        #print("Series are Stationary. P_value is: {}".format(p_value))
        #print("--------------------------------------------------------")
        return True
    else:
        #print("Series are not Stationary. P_value is: {}".format(p_value))
        #print("--------------------------------------------------------")
        return False



def cointegration_regression(df_price_X,df_price_Y):
    X = np.array(df_price_X).reshape(-1,1)
    Y = np.array(df_price_Y).reshape(-1,1)
    model = LinearRegression()
    model_fit = model.fit(X,Y)
    beta_value = model_fit.coef_[0][0]
    prediction = model_fit.predict(X)
    residual = (Y-prediction)
    residual = list(residual.ravel())
    return beta_value,residual

def check_cointegration(df_price_X,df_price_Y):
        beta_v,residual_v = cointegration_regression(df_price_X,df_price_Y)
        adf_test_value = adf_test(residual_v)
        return adf_test_value,beta_v

def double_check_cointegration_i(_df_price_merge):
    adf_test_1,beta_1 = check_cointegration(_df_price_merge["Log_price_1"],_df_price_merge["Log_price_2"])
    adf_test_2,beta_2 = check_cointegration(_df_price_merge["Log_price_2"],_df_price_merge["Log_price_1"])

    if adf_test_1 == True and adf_test_2 == True:
        return True
    else:
       
        return False


def rolling_cointegration(id_1,id_2,industry,periodicity,minimum_years):
    #HERE YOU MUST ADD A CONDITION TO CHECK IF THE DATASET ALREADY EXISTS...
    

    ## Update for new IDS

    df_price_1 = pd.read_sql("SELECT ID_INSTRUMENT,Data,Adj_Close FROM PriceIndex WHERE ID_INSTRUMENT = {}".format(id_1),acc_engine,index_col = "Data").sort_index()
    df_price_2 = pd.read_sql("SELECT ID_INSTRUMENT,Data,Adj_Close FROM PriceIndex WHERE ID_INSTRUMENT = {}".format(id_2),acc_engine,index_col = "Data").sort_index()
    asset_index = pd.read_sql("SELECT ID_INSTRUMENT,Symbol FROM AssetIndex",acc_engine)
    symbol_1 =asset_index[asset_index["ID_INSTRUMENT"]== id_1]["Symbol"].values[0]
    symbol_2 =asset_index[asset_index["ID_INSTRUMENT"]== id_2]["Symbol"].values[0]
    pair_name = symbol_1+"/"+symbol_2
    df_price_1 = df_price_1[["Adj_Close"]]
    df_price_2 = df_price_2[["Adj_Close"]]
    df_price_merge = df_price_1.merge(df_price_2,how = "inner", right_index =True, left_index = True)
    df_price_merge.rename(columns = {"Adj_Close_x": "Adj_Close_1","Adj_Close_y": "Adj_Close_2"},inplace = True)
    df_price_merge["Log_price_1"] = np.log2(df_price_merge["Adj_Close_1"])
    df_price_merge["Log_price_2"] = np.log2(df_price_merge["Adj_Close_2"])

    df_cointegration_exists = pd.read_sql("SELECT * FROM CointegrationIndex WHERE Pair= '{}'".format(pair_name),acc_engine).sort_values("Data")

    if periodicity == "D":
        rolling_window = minimum_years*250
        period_range = 250
    elif periodicity == "W":
        df_price_merge = df_price_merge.resample(periodicity).last()
        rolling_window = minimum_years*52
        period_range = 52
    else:
        pass

    if df_cointegration_exists.empty:

        new_range = len(df_price_merge) - rolling_window
        df_index = list(df_price_merge.index.values)[-(new_range):]
        is_cointegrated = []
        for i in range(new_range):
            df_ind = df_price_merge.iloc[1:rolling_window+i+1]
            coint_v = double_check_cointegration_i(df_ind)
            is_cointegrated.append(coint_v)

        df_coint_final = pd.DataFrame(is_cointegrated,index = df_index)
        df_coint_final.index.name = "Data"
        df_coint_final.rename(columns = {0:"Valor"},inplace = True)
        df_coint_final["Pair"] = pair_name
        df_coint_final["Industry"] = industry
        df_coint_final["Valor"] = df_coint_final["Valor"].astype(str)
        df_coint_final.to_sql("CointegrationIndex",acc_engine,if_exists = "append")

    else:
        index_position = df_price_merge.index.get_loc(df_price_merge[df_price_merge.index == df_cointegration_exists["Data"].iloc[-1]].index[0])
        index_position = index_position -rolling_window
        df_price_merge = df_price_merge.iloc[index_position+1:]
        if len(df_price_merge) == rolling_window:
            pass
        else:
            new_range = len(df_price_merge) - rolling_window
            df_index = list(df_price_merge.index.values)[-(new_range):]
            is_cointegrated = []
            for i in range(new_range):
                df_ind = df_price_merge.iloc[1:rolling_window+i+1]
                coint_v = double_check_cointegration_i(df_ind)
                is_cointegrated.append(coint_v)
            df_coint_final = pd.DataFrame(is_cointegrated,index = df_index)
            df_coint_final.index.name = "Data"
            df_coint_final.rename(columns = {0:"Valor"},inplace = True)
            df_coint_final["Pair"] = pair_name
            df_coint_final["Industry"] = industry
            df_coint_final["Valor"] = df_coint_final["Valor"].astype(str)
            df_coint_final.to_sql("CointegrationIndex",acc_engine,if_exists = "append")


def cointegration_analysis(id_1,id_2,industry,minimum_years,periodicity,is_rolling_or_expanding):
    df_price_1 = pd.read_sql("SELECT ID_INSTRUMENT,Data,Adj_Close FROM PriceIndex WHERE ID_INSTRUMENT = {}".format(id_1),acc_engine,index_col = "Data").sort_index()
    df_price_2 = pd.read_sql("SELECT ID_INSTRUMENT,Data,Adj_Close FROM PriceIndex WHERE ID_INSTRUMENT = {}".format(id_2),acc_engine,index_col = "Data").sort_index()
    asset_index = pd.read_sql("SELECT ID_INSTRUMENT,Symbol FROM AssetIndex",acc_engine)
    symbol_1 =asset_index[asset_index["ID_INSTRUMENT"]== id_1]["Symbol"].values[0]
    symbol_2 =asset_index[asset_index["ID_INSTRUMENT"]== id_2]["Symbol"].values[0]
    pair_name = symbol_1+"/"+symbol_2
    df_price_1 = df_price_1[["Adj_Close"]]
    df_price_2 = df_price_2[["Adj_Close"]]
    df_price_merge = df_price_1.merge(df_price_2,how = "inner", right_index =True, left_index = True)
    df_price_merge.rename(columns = {"Adj_Close_x": "Adj_Close_1","Adj_Close_y": "Adj_Close_2"},inplace = True)
    df_price_merge["Log_price_1"] = np.log2(df_price_merge["Adj_Close_1"])
    df_price_merge["Log_price_2"] = np.log2(df_price_merge["Adj_Close_2"])


    if periodicity == "D":
        rolling_window = minimum_years*250
        period_range = 250
    elif periodicity == "W":
        df_price_merge = df_price_merge.resample(periodicity).last()
        rolling_window = minimum_years*52
        period_range = 52
    else:
        pass


    # Engle Granger Test
    # Run the Linear Regression
    # Run a ADF on the residuals of the Linear regression
    # If stationary, the series are cointegrated.
    # As a rule of heuristics, use the each serie as dependent and independent variable.


    def double_check_cointegration(_df_price_merge):
        adf_test_1,beta_1 = check_cointegration(_df_price_merge["Log_price_1"],_df_price_merge["Log_price_2"])
        adf_test_2,beta_2 = check_cointegration(_df_price_merge["Log_price_2"],_df_price_merge["Log_price_1"])

        if adf_test_1 == True and adf_test_2 == True:
            print("Series for pair {} are cointegrated".format(pair_name))
            print("The Coefficient of Integration is {} for symbol {} as the INDEPENDENT variable".format(beta_1,symbol_1))
            print("--------------------------------------------------------")
            return True
        else:
            print("Series for pair {} are not cointegrated".format(pair_name))
            print("--------------------------------------------------------")
            return False
    


    pair_cointegrated = str(double_check_cointegration(df_price_merge))
    current_date = datetime.now().strftime("%d/%m/%Y")
    df_check_cointegration_exists = pd.read_sql("SELECT * FROM CointegrationCheck WHERE Pair = '{}' AND Date_Appended = '{}'".format(pair_name,current_date),acc_engine)
    if df_check_cointegration_exists.empty:
        df_check_cointegration = pd.DataFrame([[pair_name,pair_cointegrated,current_date]],columns =["Pair","is_Cointegrated","Date_Appended"])
        df_check_cointegration.set_index("Pair",inplace = True)
        #df_check_cointegration["Date_Appended"] = pd.to_datetime(df_check_cointegration["Date_Appended"],format = "%d/%m/%Y")
        df_check_cointegration["Industry"] = industry
        df_check_cointegration.to_sql("CointegrationCheck",acc_engine,if_exists = "append")
        df_check_cointegration.to_sql("CointegrationCheck",acc_engine,if_exists = "append")
    else:
        pass


    return pair_cointegrated


def pair_to_id(pair_v):
    df_asset_index = pd.read_sql("SELECT * FROM AssetIndex",acc_engine)
    ids_v = pair_v.split("/")
    id_1 = df_asset_index[df_asset_index["Symbol"] == ids_v[0]]["ID_INSTRUMENT"].values[0]
    id_2 = df_asset_index[df_asset_index["Symbol"] == ids_v[1]]["ID_INSTRUMENT"].values[0]
    return id_1,id_2

def cointegration_spread_betas(pair_v,period_range,zscore_est_period,is_rolling_or_expanding):
    id_1,id_2 = pair_to_id(pair_v)
    df_price_1 = pd.read_sql("SELECT * FROM PriceIndex WHERE ID_INSTRUMENT = {}".format(id_1),acc_engine,index_col = "Data").sort_index()
    df_price_2 = pd.read_sql("SELECT * FROM PriceIndex WHERE ID_INSTRUMENT = {}".format(id_2),acc_engine,index_col = "Data").sort_index()
    df_price_1 = df_price_1[["Adj_Close"]]
    df_price_2 = df_price_2[["Adj_Close"]]
    df_price_merge = df_price_1.merge(df_price_2,how = "inner", right_index =True, left_index = True)
    df_price_merge.rename(columns = {"Adj_Close_x": "Adj_Close_1","Adj_Close_y": "Adj_Close_2"},inplace = True)
    df_price_merge["Log_price_1"] = np.log2(df_price_merge["Adj_Close_1"])
    df_price_merge["Log_price_2"] = np.log2(df_price_merge["Adj_Close_2"])
    minimum_rolling_period = period_range*1
    len_range = len(df_price_merge) - minimum_rolling_period
    index_values = df_price_merge.iloc[-len_range-1::].index.values
    cointegration_betas = []
    spread_list = []
    if is_rolling_or_expanding == "rolling":
        for i in range(len_range):
            df_price_merge_ind = df_price_merge.iloc[0+i:minimum_rolling_period+i]
            beta_value, residuals = cointegration_regression(df_price_merge_ind["Log_price_1"],df_price_merge_ind["Log_price_2"])
            cointegration_betas.append(beta_value)
            spread_list.append(residuals[-1])

        df_cointegrated_betas = pd.DataFrame([cointegration_betas,spread_list,index_values]).T
        df_cointegrated_betas.rename(columns = {0:"Cointegration_Beta",1:"Spread_Value", 2:"Data"},inplace = True)
        df_cointegrated_betas.set_index("Data",inplace=True)
        


    elif is_rolling_or_expanding == "expanding":
        for i in range(len_range+1):
            df_price_merge_ind = df_price_merge.iloc[0:minimum_rolling_period+i]
            beta_value, residuals = cointegration_regression(df_price_merge_ind["Log_price_1"],df_price_merge_ind["Log_price_2"])
            cointegration_betas.append(beta_value)
            spread_list.append(residuals[-1])


        
        df_cointegrated_betas = pd.DataFrame([cointegration_betas,spread_list,index_values]).T
        df_cointegrated_betas.rename(columns = {0:"Cointegration_Beta",1:"Spread_Value", 2:"Data"},inplace = True)
        df_cointegrated_betas.set_index("Data",inplace=True)
        #scaler = StandardScaler()
        #df_cointegrated_betas["zscore_spread"] = scaler.fit_transform(df_cointegrated_betas[["Spread_Value"]])

    else:
        pass
    std_v_list = []
    new_range = len(df_cointegrated_betas) - zscore_est_period
    index_v = df_cointegrated_betas.index.values[-(new_range)::]

    for i in range(new_range):
        df_cointegrated_betas_ind = df_cointegrated_betas.iloc[0:zscore_est_period+i+1]     
        std_v = (df_cointegrated_betas_ind.iloc[-1]["Spread_Value"]- df_cointegrated_betas_ind["Spread_Value"].mean()) / df_cointegrated_betas_ind["Spread_Value"].std()
        std_v_list.append(std_v)
        

    df_zscore_spread = pd.DataFrame(std_v_list,columns = ["zscore_spread"])
    df_zscore_spread["Data"] = index_v
    df_zscore_spread.set_index("Data",inplace = True)
    df_cointegrated_betas = df_zscore_spread.merge(df_cointegrated_betas, how = "inner",right_index = True, left_index = True)
    return df_cointegrated_betas

#cointegration_spread_betas("MU/NVDA",250,125,"expanding")

