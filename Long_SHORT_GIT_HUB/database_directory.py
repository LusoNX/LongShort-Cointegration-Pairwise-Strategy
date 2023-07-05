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


## Directory for the databse
def where_am_i(directory):
    #r = (r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
    #    r'DBQ={}\AssetDataBase.accdb;').format(directory)

    r = fr"DRIVER={{Microsoft Access Driver (*.mdb, *.accdb)}};DBQ={directory}\AssetDataBase.accdb;"

    return r


## Directory for the image download
def image_directory():
    your_directory_rets = "YOUR_DIRECTORY" + r"\RETS"
    your_directory_beta = "YOUR_DIRECTORY" + r"\BETA"
    return your_directory_rets,your_directory_beta

def main():
    your_directory = r"C:\Users\Luan\Desktop\LONG_SHORT_strategies\Database"
    r = where_am_i("YOUR DIRECTORY")
    return r