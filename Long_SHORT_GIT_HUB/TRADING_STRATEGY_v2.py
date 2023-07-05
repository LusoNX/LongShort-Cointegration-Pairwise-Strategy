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
import long_short_strategy
conn_str = database_directory.main()
your_directory_rets,your_directory_beta = database_directory.image_directory()
cnn_url = f"access+pyodbc:///?odbc_connect={urllib.parse.quote_plus(conn_str)}"
acc_engine = create_engine(cnn_url)

## Residual Momentum W/L
def winners_losers(industry):
    benchmark = pd.read_sql("SELECT Benchmark_Name,Industry FROM BenchmarkIndex WHERE Industry = '{}'".format(industry),acc_engine)["Benchmark_Name"].values[0]
    df_momentum_rank = pd.read_sql("SELECT * FROM RankTable WHERE Rank_Type = '{}' AND Benchmark = '{}'".format("Momentum",benchmark),acc_engine).sort_values("Data")
    unique_dates = list(df_momentum_rank["Data"].unique())
    list_df = []

    for i in range(len(unique_dates)):
        df_ind = df_momentum_rank[df_momentum_rank["Data"] == unique_dates[i]]
        df_ind = df_ind.copy()
        #win_los = pd.cut(df_ind["Rank_Value"],bins = [0,df_ind["Rank_Value"].median(),df_ind["Rank_Value"].max()],labels = ["W","L"])
        df_ind["Winner_Loser"] = pd.cut(df_ind["Rank_Value"],bins = [0,df_ind["Rank_Value"].median(),df_ind["Rank_Value"].max()],labels = ["W","L"])
        df_ind = df_ind.sort_values("ID_INSTRUMENT")
        list_df.append(df_ind)
    df_momentum = pd.concat(list_df)
    df_momentum = df_momentum.sort_values("Data")
    return df_momentum

## Full Momentum W/L
def winners_losers_cryptocurrency(_industry):
    df_index = pd.read_sql("SELECT * FROM AssetIndex WHERE Industry = '{}'".format(_industry),acc_engine).sort_values("ID_INSTRUMENT")
    unique_ids = list(df_index["ID_INSTRUMENT"].unique())
    df_id = pd.read_sql("SELECT ID_INSTRUMENT,Data,Adj_Close FROM PriceIndex WHERE ID_INSTRUMENT = {}".format(unique_ids[0]),acc_engine).sort_values("Data")
    df_merge = pd.DataFrame(df_id.iloc[-(len(df_id)-250)::]["Data"])
    for i in range(len(unique_ids)):
        df_price = pd.read_sql("SELECT ID_INSTRUMENT,Data,Adj_Close FROM PriceIndex WHERE ID_INSTRUMENT = {}".format(unique_ids[i]),acc_engine).sort_values("Data")
        df_price[unique_ids[i]] = df_price["Adj_Close"].pct_change(250)
        df_price.dropna(inplace = True)
        df_price = df_price[["Data",unique_ids[i]]]
        df_merge = df_merge.merge(df_price[[unique_ids[i],"Data"]],on = "Data",how = "outer")
    
    def win_los(row,value):
        if row > value:
            return "W"
        else:
            return "L"

    df_merge.set_index("Data",inplace = True)
    df_list = []
    for i in range(len(df_merge)):
        date_v = df_merge.index.values[i]
        df_merge_ind = df_merge[df_merge.index == date_v]  
        df_merge_ind = df_merge_ind.dropna(axis = 1)
       

        df_merge_ind = df_merge_ind.T     
        df_merge_ind['Winner_Loser'] = np.where(df_merge_ind[date_v] > df_merge_ind[date_v].median(), 'W', 'L')

        df_merge_ind.index.name = "ID_INSTRUMENT"
        df_merge_ind["Data"] = date_v
        df_merge_ind = df_merge_ind[["Winner_Loser","Data"]]
        df_list.append(df_merge_ind)

    df_final_merge = pd.concat(df_list)
    df_final_merge.reset_index(inplace = True)
    df_final_merge = df_final_merge.sort_values("Data")
    return df_final_merge
    

## Trading Strategy input implementation
def trading_signal_pair_STRAT_2(id_1,id_2,beta_type,momentum_factor,momentum_bench,correlation_threshold,restriction_type,zscore_type,zscore_type_2,zscore_close,ma_zscore_1_thr,ma_zscore_2_thr,close_thr,expanding_rolling,weight_adj,trim_beta):
    asset_index = pd.read_sql("SELECT * FROM AssetIndex",acc_engine)
    asset_industry = asset_index[asset_index["ID_INSTRUMENT"] == id_1]["Industry"].values[0]
    symbol_1 = asset_index[asset_index["ID_INSTRUMENT"] == id_1]["Symbol"].values[0]
    symbol_2 = asset_index[asset_index["ID_INSTRUMENT"] == id_2]["Symbol"].values[0]
    symbol_pair = symbol_1 +"/" + symbol_2
    inverse_symbol_pair = symbol_2 +"/" + symbol_1


    df_cointegration = pd.read_sql("SELECT * FROM CointegrationIndex WHERE Pair = '{}'".format(symbol_pair),acc_engine).sort_values("Data")
    if df_cointegration.empty:
        df_cointegration  = pd.read_sql("SELECT * FROM CorrelationIndex WHERE Pair = '{}'".format(inverse_symbol_pair),acc_engine).sort_values("Data")
    else:
        pass


    def trim_weight(row):
        if row >=0.75:
            return 0.75
        elif row <= 0.25:
            return 0.25
        else:
            return row

    df_cointegration.set_index("Data",inplace = True)
    df_beta_weights = long_short_strategy.beta_neutrality(id_1,id_2,beta_type)
    if trim_beta == True:
        df_beta_weights["weight_1"] = df_beta_weights["weight_1"].apply(trim_weight)
        df_beta_weights["weight_2"] = df_beta_weights["weight_2"].apply(trim_weight)
    else:
        pass
  
    
    df_cointegration = df_cointegration[~df_cointegration.index.duplicated(keep='first')]
    df_zscore_spread = correlation_stats.cointegration_spread_betas(symbol_pair,250,125,expanding_rolling) # 250 is 1 year
    df_merge = df_zscore_spread.copy()
    df_merge.fillna(method = "ffill",inplace = True)
    if weight_adj == "equal_weight":
        df_merge["weight_1"] = 0.5
        df_merge["weight_2"] = 0.5
    elif weight_adj == "beta_weight":
        df_merge = df_merge.merge(df_beta_weights[["weight_1","weight_2"]],how = "inner", right_index = True, left_index = True)
    else:
        pass

    df_merge.reset_index(inplace = True)
    df_merge = df_merge[df_merge.Data.dt.weekday < 5]
    df_merge.set_index("Data",inplace = True)
    df_merge.sort_index(inplace = True)
    pair_signal = {}
    df_merge.reset_index(inplace = True)
    df_merge.fillna(method = "ffill",inplace = True)
    control_variable = "Close Trade"

    if momentum_factor == True:
        #df_momentum = winners_losers(momentum_bench)
    
        df_momentum = winners_losers_cryptocurrency(asset_industry)

        df_momentum_id_1 = df_momentum[df_momentum["ID_INSTRUMENT"] == id_1].sort_values("Data")
        df_momentum_id_2 = df_momentum[df_momentum["ID_INSTRUMENT"] == id_2].sort_values("Data")
        df_momentum_id_1.rename(columns = {"Winner_Loser":"Pair_1_WL"},inplace = True)
        df_momentum_id_2.rename(columns = {"Winner_Loser":"Pair_2_WL"},inplace = True)
        df_merge = df_merge.merge(df_momentum_id_1[["Data","Pair_1_WL"]],on = "Data",how = "outer")
        df_merge.dropna(inplace = True)
        df_merge = df_merge.sort_values("Data")
        df_merge = df_merge.merge(df_momentum_id_2[["Data","Pair_2_WL"]],on = "Data",how = "outer")
        df_merge = df_merge.sort_values("Data")
        df_merge.dropna(inplace = True)

        pair_signal[df_merge["Data"].iloc[0]] = ""
        for i in range(len(df_merge)):
            df_merge_ind = df_merge.iloc[i]
            date_v = df_merge_ind["Data"]
            if i >= 1:
                df_merge_old = df_merge.iloc[i-1]
                date_old = df_merge_old["Data"]
            else:
                df_merge_old = df_merge.iloc[i]
                date_old = df_merge_old["Data"]
            
            
            if pair_signal[date_old] == "":
                #if df_merge_ind["Valor"] == "True":
                if restriction_type == "single":
                    if df_merge_ind[zscore_type] >= ma_zscore_1_thr and df_merge_ind["Pair_1_WL"] == "L" and df_merge_ind["Pair_2_WL"] == "W":
                        pair_signal[date_v] = "SHORT"
                    elif df_merge_ind[zscore_type] <= (-1)*ma_zscore_1_thr and df_merge_ind["Pair_1_WL"] == "W" and df_merge_ind["Pair_2_WL"] == "L":
                        pair_signal[date_v] = "LONG"
                    else:
                        pair_signal[date_v] = ""

                elif restriction_type == "double":
                    if df_merge_ind[zscore_type] >= ma_zscore_1_thr and df_merge_ind["Pair_1_WL"] == "L" and df_merge_ind["Pair_2_WL"] == "W":
                        pair_signal[date_v] = "SHORT"
                    elif df_merge_ind[zscore_type] <= (-1)*ma_zscore_1_thr and df_merge_ind[zscore_type_2] <= (-1)*ma_zscore_2_thr and df_merge_ind["Pair_1_WL"] == "W" and df_merge_ind["Pair_2_WL"] == "L":
                        pair_signal[date_v] = "LONG"

                    else:
                        pair_signal[date_v] = ""
                else:
                    pass
                #else:
                #        pair_signal[date_v] = ""


            elif pair_signal[date_old] == "LONG":
                #if df_merge_ind["Valor"] == "True":
                if restriction_type == "single":
                    if df_merge_ind[zscore_type] >= close_thr and df_merge_ind["Pair_1_WL"] == "L" and df_merge_ind["Pair_2_WL"] == "W":
                        pair_signal[date_v] = "SHORT"
                    elif df_merge_ind[zscore_close] <= (-1)*close_thr and df_merge_ind["Pair_1_WL"] == "W" and df_merge_ind["Pair_2_WL"] == "L":
                            pair_signal[date_v] = "LONG"
                    else:
                        pair_signal[date_v] = ""
                elif restriction_type  == "double":
                    if df_merge_ind[zscore_close] >= close_thr:
                        pair_signal[date_v] = "SHORT"
                    elif df_merge_ind[zscore_type] <= (-1)*ma_zscore_1_thr:
                            pair_signal[date_v] = "LONG"
                    else:
                        pair_signal[date_v] = ""
                

            elif pair_signal[date_old] == "SHORT":
                #if df_merge_ind["Valor"] == "True":
                if restriction_type == "single":
                    if df_merge_ind[zscore_close] >= close_thr and df_merge_ind["Pair_1_WL"] == "L" and df_merge_ind["Pair_2_WL"] == "W":
                        pair_signal[date_v] = "SHORT"
                    elif df_merge_ind[zscore_type] <= (-1)*close_thr and df_merge_ind["Pair_1_WL"] == "W" and df_merge_ind["Pair_2_WL"] == "L":
                            pair_signal[date_v] = "LONG"
                    else:
                        pair_signal[date_v] = ""

                elif restriction_type == "double":
                    if df_merge_ind[zscore_type] >= ma_zscore_1_thr and df_merge_ind[zscore_type_2] >= ma_zscore_2_thr and df_merge_ind["Pair_1_WL"] == "W" and df_merge_ind["Pair_2_WL"] == "L":
                        pair_signal[date_v] = "SHORT"
                    elif df_merge_ind[zscore_close] <= close_thr:
                            pair_signal[date_v] = "LONG"
                    else:
                        pair_signal[date_v] = ""

          
            else:
                pair_signal[date_v] = ""    

    else:
        df_merge = df_merge.sort_values("Data")
        df_merge.dropna(inplace = True)
        pair_signal[df_merge["Data"].iloc[0]] = ""

        for i in range(len(df_merge)):
            df_merge_ind = df_merge.iloc[i]
            date_v = df_merge_ind["Data"]
            if i >= 1:
                df_merge_old = df_merge.iloc[i-1]
                date_old = df_merge_old["Data"]
            else:
                df_merge_old = df_merge.iloc[i]
                date_old = df_merge_old["Data"]
            
            
            if pair_signal[date_old] == "":
                if restriction_type == "single":
                    if df_merge_ind[zscore_type] >= ma_zscore_1_thr:
                        pair_signal[date_v] = "SHORT"
                    elif df_merge_ind[zscore_type] <= (-1)*ma_zscore_1_thr :
                        pair_signal[date_v] = "LONG"
                    else:
                        pair_signal[date_v] = ""

                elif restriction_type == "double":
                    if df_merge_ind[zscore_type] >= ma_zscore_1_thr :
                        pair_signal[date_v] = "SHORT"
                    elif df_merge_ind[zscore_type] <= (-1)*ma_zscore_1_thr and df_merge_ind[zscore_type_2] <= (-1)*ma_zscore_2_thr :
                        pair_signal[date_v] = "LONG"

                    else:
                        pair_signal[date_v] = ""
                else:
                    pair_signal[date_v] = ""
            

            elif pair_signal[date_old] == "LONG":
                
                if restriction_type == "single":
                    if df_merge_ind[zscore_close] >= close_thr:
                        pair_signal[date_v] = "SHORT"
                    elif df_merge_ind[zscore_type] <= (-1)*ma_zscore_1_thr:
                            pair_signal[date_v] = "LONG"
                    else:
                        pair_signal[date_v] = ""
                elif restriction_type  == "double":
                    if df_merge_ind[zscore_close] >= close_thr:
                        pair_signal[date_v] = "SHORT"
                    elif df_merge_ind[zscore_type] <= (-1)*ma_zscore_1_thr:
                            pair_signal[date_v] = "LONG"
                    else:
                        pair_signal[date_v] = ""
                

            elif pair_signal[date_old] == "SHORT":

                #if df_merge_ind["Correlation"] >= correlation_threshold:
                if restriction_type == "single":
                    if df_merge_ind[zscore_type] >= ma_zscore_1_thr :
                        pair_signal[date_v] = "SHORT"
                    elif df_merge_ind[zscore_close] <= (-1)*close_thr:
                            pair_signal[date_v] = "LONG"
                    else:
                        pair_signal[date_v] = ""

                elif restriction_type == "double":
                    if df_merge_ind[zscore_type] >= ma_zscore_1_thr and df_merge_ind[zscore_type_2] >= ma_zscore_2_thr :
                        pair_signal[date_v] = "SHORT"
                    elif df_merge_ind[zscore_close] <= close_thr:
                            pair_signal[date_v] = "LONG"
                    else:
                        pair_signal[date_v] = ""   
            
            else:
                pair_signal[date_v] = ""

              

    df_trades = pd.DataFrame(pair_signal,index = [1]).T
    df_trades.index.name = "Data"
    df_merge.set_index("Data",inplace = True)
    df_trades.rename(columns = {1:"SIGNAL"},inplace = True)
    df_merged_final = df_trades.merge(df_merge, how = "inner", right_index = True, left_index = True)
    df_merged_final["Pair"] = symbol_pair
    return df_merged_final
       

## Backtest of the Strategy
def backtest_strategy(id_1,id_2,confirmation_type,std_OPEN,std_CONFIRM,std_CLOSE,momentum_reversal,industry_momentum,include_momentum,expanding_rolling,weight_adj,trim_beta):
    df_trade_signal = trading_signal_pair_STRAT_2(id_1,id_2,"US_SP500",include_momentum,industry_momentum,0,confirmation_type,"zscore_spread","zscore_spread","zscore_spread",std_OPEN,std_CONFIRM,std_CLOSE,expanding_rolling,weight_adj,trim_beta)
    df_price_1 = pd.read_sql("SELECT Data, Adj_Close,ID_INSTRUMENT FROM PriceIndex WHERE ID_INSTRUMENT = {}".format(id_2),acc_engine,index_col = "Data").sort_index()
    df_price_2 = pd.read_sql("SELECT Data, Adj_Close,ID_INSTRUMENT FROM PriceIndex WHERE ID_INSTRUMENT = {}".format(id_1),acc_engine,index_col = "Data").sort_index()
    df_price_1.rename(columns = {"Adj_Close":"pair_1_p"},inplace = True)
    df_price_2.rename(columns = {"Adj_Close":"pair_2_p"},inplace = True)
    df_price_merge = df_price_1[["pair_1_p"]].merge(df_price_2[["pair_2_p"]],how = "inner", right_index = True, left_index = True).sort_index()
    df_price_merge.reset_index(inplace = True)
    df_trade_signal.reset_index(inplace = True)
    df_merge_final = df_trade_signal.merge(df_price_merge,on ="Data",how='outer')
    df_merge_final = df_merge_final.sort_values("Data")
    df_merge_final["pair_1_r"] = df_merge_final["pair_1_p"].pct_change()
    df_merge_final["pair_2_r"] = df_merge_final["pair_2_p"].pct_change()
    df_merge_final.fillna(method = "ffill",inplace = True)
    dict_returns = {}
    momentum_reversal_dict = {"momentum":1,"mean_reversion":-1}
    mom_rev_value = momentum_reversal_dict[momentum_reversal]
    #l_s_signals = df_merge_final["SIGNAL"].value_counts()
    pair_name = df_trade_signal.iloc[0]["Pair"]
    df_signals = df_merge_final[["Data","SIGNAL"]].copy()
    df_signals.set_index("Data",inplace = True)
    for i in range(len(df_merge_final)-1):
        df_merge_ind = df_merge_final.iloc[i]
        date_v = df_merge_final.iloc[i+1]["Data"]
        if df_merge_ind["SIGNAL"] == "SHORT":
            ret_1 = df_merge_final.iloc[i+1]["pair_1_r"]*mom_rev_value*df_merge_ind["weight_1"]
            ret_2 =df_merge_final.iloc[i+1]["pair_2_r"]*(-1)*mom_rev_value*df_merge_ind["weight_2"]
            ret_values = [ret_1,ret_2]
            dict_returns[date_v] = ret_values
        elif df_merge_ind["SIGNAL"] == "LONG":
            ret_1 = df_merge_final.iloc[i+1]["pair_1_r"]*(-1)*mom_rev_value*df_merge_ind["weight_1"]
            ret_2 =df_merge_final.iloc[i+1]["pair_2_r"]*mom_rev_value*df_merge_ind["weight_2"]
            ret_values = [ret_1,ret_2]
            dict_returns[date_v] = ret_values
        else:
            dict_returns[date_v] = [0,0]

    df_returns = pd.DataFrame(dict_returns).T
    df_returns.rename(columns = {0:"Pair_1_ret", 1:"Pair_2_ret"},inplace = True)
    df_returns.index.name = "Data"
    strat_ret_name = "Strategy_ret_{}".format(pair_name)
    df_returns[strat_ret_name] = (df_returns["Pair_1_ret"]+df_returns["Pair_2_ret"])
    #Create a second dataframe for a special backtest.
    df_signals_pair = df_signals.copy()
    df_signals_pair.rename(columns = {"SIGNAL":"SIGNAL_{}".format(pair_name)},inplace = True)
    df_ret_pair = df_returns[[strat_ret_name]].copy()
    
    df_signal_ret = df_signals_pair.merge(df_ret_pair, how = "inner",right_index = True, left_index = True)
    return df_returns,df_signals,df_signals_pair,df_signal_ret,df_merge_final

#backtest_strategy(1,5,"momentum")

def port_weight_overtime(DF,pair_name):
    df = DF.copy()
    df_weights = pd.DataFrame(index = df.index)
    for i in pair_name:
        df_ind = df.filter(regex=i)
        df_weights[i] = df_ind.sum(axis = 1)
    
    return df_weights
    
## Multiple backtests (over all the pairs traded)
def multiple_backtests(confirmation_type,std_open,std_confirm,std_close,industry,exclude_crypto,start_time,last_date,include_momentum,expanding_rolling,weight_adj,trim_beta):
    index_list = list(pd.read_sql("SELECT ID_INSTRUMENT,Instrument_Type FROM AssetIndex WHERE Instrument_Type ='{}'".format("Index"),acc_engine)["ID_INSTRUMENT"].unique())
    df_cointegrated = pd.read_sql("SELECT * FROM CointegrationCheck",acc_engine)
    #df_cointegrated = pd.read_sql("SELECT * FROM CointegrationIndex",acc_engine)
    df_cointegrated_original = df_cointegrated.copy()
    if industry == "all":
        pass
    else:
        df_cointegrated = df_cointegrated[df_cointegrated["Industry"] == industry]

    if exclude_crypto == True:
        df_cointegrated = df_cointegrated[df_cointegrated["Industry"] != "Cryptocurrency"]
    else:
        pass

    pairs_symbol = list(df_cointegrated[df_cointegrated["is_Cointegrated"] == "True"]["Pair"].unique())
    pairs_symbol = [ x for x in pairs_symbol if "Index" not in x ]
    pairs_symbol_original = pairs_symbol
    pairs_symbol = [list(x.split("/")) for x in pairs_symbol]
    df_index = pd.read_sql("SELECT * FROM AssetIndex",acc_engine)
    date_range = pd.date_range(start =start_time, end = last_date,freq = "D")
    dict_strat_ret = {}
    signal_dict = {}


    def signal_dummy(row):
        if row == "LONG":
            return 1
        elif row == "SHORT":
            return -1
        else:
            return 0

    df_port = pd.DataFrame(date_range,columns = ["Data"])
    if len(pairs_symbol) == 0:
        return pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
        print("No pairs are cointegrated for this industry: {}".format(industry))
        pass
    else:
        pair_name_list = []
        for x in range(len(pairs_symbol)):
            symbol_pair = pairs_symbol[x]
            pair_name = symbol_pair[0] +"/"+symbol_pair[1]
            print("Pairs included in the strategy:", symbol_pair[0] +"/" +symbol_pair[1])
            id_1 = df_index[df_index["Symbol"] == symbol_pair[0]]["ID_INSTRUMENT"].values[0]
            id_2 = df_index[df_index["Symbol"] == symbol_pair[1]]["ID_INSTRUMENT"].values[0]
            industry_momentum = df_cointegrated_original[df_cointegrated_original["Pair"] == pairs_symbol_original[x]]["Industry"].values[0] #used for momentum
            df_rets,df_signals,df_signals_pair,df_signal_ret,df_merge_final = backtest_strategy(id_1,id_2,confirmation_type,std_open,std_confirm,std_close,"mean_reversion",industry_momentum,include_momentum,expanding_rolling,weight_adj,trim_beta)
            df_signal_ret.reset_index(inplace = True)
            #portion for the portfolio creation 
            df_merge_final.rename(columns = {"weight_1":symbol_pair[0],"weight_2":symbol_pair[1]},inplace = True)
            df_merge_final["SIGNAL_D"] = df_merge_final["SIGNAL"].apply(signal_dummy)
            df_merge_final[symbol_pair[0]] = df_merge_final[symbol_pair[0]]*df_merge_final["SIGNAL_D"]
            df_merge_final[symbol_pair[1]] = df_merge_final[symbol_pair[1]]*df_merge_final["SIGNAL_D"]*(-1)
            pair_name_list.append(symbol_pair[0])
            pair_name_list.append(symbol_pair[1])


            df_merge_port = df_merge_final[["Data",symbol_pair[0],symbol_pair[1]]]
            df_merge_port.set_index("Data",inplace = True)
            # Shift 1 in final value, because signal is given priorly to the opening/closure of the position
            df_merge_port = df_merge_port.shift(1)
            df_merge_port.reset_index(inplace = True)

            df_merge_port = df_merge_port.rename(columns={df_merge_port.columns[1]: str(df_merge_port.columns[1])+"_"+str(x), df_merge_port.columns[2]: str(df_merge_port.columns[2])+"_"+str(x)})
            df_port = df_port.merge(df_merge_port,on = "Data", how = "outer")


            if x == 0:
                df_merge = df_signal_ret.copy()
            else:
                df_merge = df_signal_ret.merge(df_merge,on = "Data",how="outer" ).sort_values("Data")


            df_rets.reset_index(inplace = True)
            strat_ret_name = "Strategy_ret_{}".format(pair_name)
            df_ret =df_rets[["Data","Pair_1_ret","Pair_2_ret",strat_ret_name]]
            df_ret_strat = df_ret.copy()
            df_ret_strat.set_index("Data",inplace = True)
            dict_strat_ret[pair_name] = df_ret_strat
            signal_dict[pair_name] = df_signals
           

        
        df_merge.set_index("Data",inplace = True)
        df_merge = df_merge.replace("LONG","1")
        df_merge = df_merge.replace("SHORT","1")
        df_merge = df_merge.replace("","0")
        df_merge.fillna(0,inplace = True)
        df_merge = df_merge.astype(float)
        df_merge_sig = df_merge.copy()
        df_merge_strat = df_merge.copy()
        signal_cols = [x for x in list(df_merge.columns) if "SIGNAL_" in x]
        df_merge_sig = df_merge_sig[signal_cols]
        df_merge_sig["nr_trades"] = df_merge_sig[signal_cols].sum(axis =1)
        #df_merge_sig.to_csv("merge_sig.csv",sep =";")
        main_ret_cols = list(df_merge.columns)
        main_ret_cols = [x for x in main_ret_cols if "Strategy_ret" in x]
        df_strat_ret = df_merge_strat[main_ret_cols]
        df_strat_ret = df_strat_ret.copy()
        df_strat_ret["Final_ret"] = df_strat_ret.sum(axis = 1)
        df_strat_ret =  df_strat_ret.merge(df_merge_sig[["nr_trades"]], how = "inner", right_index = True, left_index = True)
        df_port.set_index("Data",inplace = True)
        df_port = df_port.merge(df_strat_ret[["nr_trades"]],how = "inner", right_index = True, left_index = True)
        df_port["nr_trades"] = df_port["nr_trades"].shift(1)
        list_columns = list(df_port.columns)
        list_columns.remove("nr_trades")
        df_port.fillna(0,inplace = True)

        #for x in list_columns:
        #    df_port[x] = df_port[x] / (df_port["nr_trades"]*2)
        pair_name_list = np.unique(np.array(pair_name_list))
        df_port =df_port[list_columns]
        df_port_weights =port_weight_overtime(df_port,pair_name_list)
        #fig, ax = plt.subplots(nrows=1, ncols=1, facecolor="#F0F0F0")
        #for x in list(pair_name_list):
        #    ax.stackplot(df_port_weights.index, df_port_weights[x],labels = x)
        #ax.set_xlabel("Time")
        #ax.set_ylabel("Weight")
        #ax.set_ylim(bottom=-1, top=1)
        #ax.grid(which="major", color="grey", linestyle="--", linewidth=0.5)
        #ax.legend()
        #plt.show()
        

        df_strat_ret["nr_trades"] = df_strat_ret["nr_trades"].shift(1)
        df_strat_ret = df_strat_ret.iloc[1::] 
        df_strat_ret["Final_ret"] = df_strat_ret["Final_ret"]/df_strat_ret["nr_trades"]
        df_strat_ret.fillna(0,inplace = True)
        df_strat_ret_cum = (1+df_strat_ret).cumprod()
        df_strat_ret_cum["Final_ret"].plot()
        plt.title("Acc. Performance for {}".format(industry))
        plt.savefig(r"{}\FULL_DATA_RET_{}.png".format(your_directory_rets,industry))
        plt.close()
        return df_strat_ret, dict_strat_ret,signal_dict


def beta_market(df_price_X,range_window,industry):
    df_market_ret = pd.read_sql("SELECT ID_INSTRUMENT,Data,Adj_Close FROM PriceIndex WHERE ID_INSTRUMENT = {}".format(10),acc_engine,index_col = "Data").sort_index()
    df_market_ret = df_market_ret.pct_change()
    df_market_ret = df_market_ret.iloc[1::]
    df_price_X.reset_index(inplace = True)
    df_market_ret.reset_index(inplace = True)
    df_merge = df_price_X.merge(df_market_ret[["Adj_Close","Data"]],on = "Data",how = "outer")
    df_merge.set_index("Data",inplace = True)
    df_merge.dropna(inplace = True)
    new_range = len(df_merge) - range_window
    beta_list = []

    index_v = df_merge.iloc[-new_range::].index.values
    for i in range(new_range):
        df_merge_ind = df_merge.iloc[0+i+1:range_window+i+1]
        X = np.array(df_merge_ind["Adj_Close"]).reshape(-1,1)
        Y = np.array(df_merge_ind["Final_ret"]).reshape(-1,1)
        try:
            model = LinearRegression()
            model_fit = model.fit(X,Y)
            beta_value = model_fit.coef_[0][0]
        except:
            beta_value = 0
        beta_list.append(beta_value)
    df_beta = pd.DataFrame(beta_list)
    df_beta["Data"] = index_v
    df_beta.rename(columns = {0:"Beta"},inplace = True)
    df_beta.set_index("Data",inplace = True)
    df_beta.plot()
    plt.title("Beta Value for {}".format(industry))
    plt.savefig(r"{}\FULL_DATA_BETA_{}.png".format(your_directory_beta,industry))
    plt.close()
    return df_beta


## Strategy performance details. 
def strategy_performance_details(industry,strat_dict,dict_signal):
    def max_dd(DF,col_name):
        "function to calculate max drawdown"
        df = DF.copy()
        df["cum_return"] = (1 + df[col_name]).cumprod()
        df["cum_roll_max"] = df["cum_return"].cummax()
        df["drawdown"] = df["cum_roll_max"] - df["cum_return"]
        df["drawdown_pct"] = df["drawdown"]/df["cum_roll_max"]
        max_dd = df["drawdown_pct"].max()
        return max_dd

    list_of_lists = []
    column_v = ["Pair","Count","Pair_1_MIN","Pair_2_MIN","Strat_MIN","Pair_1_MAX","Pair_2_MAX","Strat_MAX","Pair_1_ACC","Pair_2_ACC","Strat_ACC","Pair_1_MaxDD","Pair_2_MaxDD","Strat_MaxDD","LONG_signals","SHORT_signals"]
    for x in strat_dict:
        x_v = x.split("/")
        pair_1 = x_v[0]
        pair_2 = x_v[1]
        pair_name = str(pair_1)+"/"+pair_2
        column_names = [pair_1,pair_2,x_v]
        df_strat = strat_dict[x]
        df_strat_des = df_strat.describe()
        min_v = df_strat.min()
        max_v = df_strat.max()        
        acc_ret = ((1+df_strat).cumprod()).iloc[-1]
        max_dd_1 = max_dd(df_strat,"Pair_1_ret")
        max_dd_2 =max_dd(df_strat,"Pair_2_ret")
        strat_ret_name = "Strategy_ret_{}".format(pair_name)
        max_dd_pair =max_dd(df_strat,strat_ret_name)
        df_signal = dict_signal[x]
        signal_count = df_signal.value_counts()
       
        try:
            long_signal = signal_count["LONG"]
        except:
            long_signal = 0
        try:
            short_signal = signal_count["SHORT"]
        except:
            short_signal = 0
        list_of_values = [x,len(df_strat),min_v["Pair_1_ret"],min_v["Pair_2_ret"],min_v[strat_ret_name],max_v["Pair_1_ret"],max_v["Pair_2_ret"],max_v[strat_ret_name],acc_ret["Pair_1_ret"],acc_ret["Pair_2_ret"],acc_ret[strat_ret_name],max_dd_1,max_dd_2,max_dd_pair,long_signal,short_signal]
        list_of_lists.append(list_of_values)

    df_strat_stats = pd.DataFrame(list_of_lists,columns = column_v)
    df_strat_stats.to_csv(r"{}\FULL_DATA_Performance_Stats_{}.csv".format(your_directory_rets,industry))
   


# Runs an entire loop on all industries to check their stats
def check_backtests():
    list_of_sectors = ["Semiconductors","Restaurants","REIT—Healthcare Facilities","Telecom and Entertainment","Bank and Capital Markets","Oil & Gas Integrated","Airlines","Software and Information","all"]
    for i in list_of_sectors:

        print("---------------------|||||-------------------------------")
        print(i)
        df_strategy_ret,dict_strat,dict_signal = multiple_backtests("single",1,1,0.5,i,True,"2016-01-01","2023-01-06",True,"expanding","beta_weight",True)
        if df_strategy_ret.empty:
            pass
        else:
            beta_market(df_strategy_ret[["Final_ret"]],250,i)
            strategy_performance_details(i,dict_strat,dict_signal)
#check_backtests()


# Evaluates multiple backtests to check the better fit, given the paramers used for the signals.
def evaluate_multiple_backtests():
    unique_industries = ["Bank and Capital Markets","Oil & Gas E&P","Semiconductors","Cryptocurrency"]
    open_std = [1,2]
    close_std = [0.25,0.5,1]
    value_type = ["single"]
    rolling_type = ["rolling","expanding"]
    column_names = ["Industry","Conf_Type","open_std","confirm_std","close_std","Med_Ret","Acc_Ret","Volat","Skew","Kurt","Max_Beta","Min_Beta"]
    list_appends = []
    for x in unique_industries:
        for y in value_type:
            for b in rolling_type:
                for i in open_std:
                        for z in close_std:
                            df_strategy_ret = multiple_backtests(y,i,0,z,x,False,"2016-01-01","2023-01-06",True,b)
                            df_strategy_ret["Acc_Ret"] = (1+df_strategy_ret["Final_ret"]).cumprod()
                            med_ret = df_strategy_ret["Final_ret"].median()
                            vol = df_strategy_ret["Final_ret"].std()
                            skew = df_strategy_ret["Final_ret"].skew()
                            kurt = df_strategy_ret["Final_ret"].kurt()
                            last_acc = df_strategy_ret.iloc[-1]["Acc_Ret"]
                            df_beta = beta_market(df_strategy_ret[["Final_ret"]],250,x)
                            beta_max = df_beta["Beta"].max()
                            beta_min = df_beta["Beta"].min()
                            value_list = [x,y,i,b,z,med_ret,last_acc,vol,skew,kurt,beta_max,beta_min]
                            list_appends.append(value_list)

    df_v = pd.DataFrame(list_appends,columns = column_names)
    return df_v

def main():
    check_backtests()
    #evaluate_multiple_backtests()

main()