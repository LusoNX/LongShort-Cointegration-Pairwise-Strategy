import pandas as pd
import sys
from sqlite3 import connect
import urllib
from sqlalchemy import create_engine
import pyodbc
import urllib

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


def createDataBase():
    conn_str = pyodbc.connect(conn_str)

    assetIndex_table = """
    CREATE TABLE AssetIndex(
    ID_INSTRUMENT INT PRIMARY KEY,
    Instrument_Type VARCHAR(50),
    ISIN VARCHAR(50),
    Name VARCHAR(150),
    Symbol VARCHAR(50),
    Industry VARCHAR(5),
    Country VARCHAR(50)
    );
    """

    statsIndex_table = """
    CREATE TABLE StatsIndex(
    ID_INSTRUMENT INT,
    Data DATETIME,
    annual_return FLOAT,
    vol_anualized FLOAT,    
    skew_anualized FLOAT,
    kurt_anualized FLOAT
    );
    """

    PriceIndex_table = """
    CREATE TABLE PriceIndex(
    ID_INSTRUMENT INT,
    Data DATETIME,
    Open FLOAT,
    High FLOAT,
    Low FLOAT,
    Close FLOAT,
    FOREIGN KEY(ID_INSTRUMENT) REFERENCES AssetIndex(ID_INSTRUMENT)
    );
    """

    correlationIndex_table = """
    CREATE TABLE CorrelationIndex(
    Data DATETIME,
    Correlation FLOAT,
    Pair VARCHAR(50)
    );
    """
    BetaIndex_table = """
    CREATE TABLE BetaIndex(
    ID_INSTRUMENT INT,
    Data DATETIME,
    Beta FLOAT,
    R2 FLOAT,
    Beta_Type VARCHAR(50),
    FOREIGN KEY(ID_INSTRUMENT) REFERENCES AssetIndex(ID_INSTRUMENT)
    );
    """

    dataRank_table = """
    CREATE TABLE RankTable(
    ID_INSTRUMENT INT,
    Data DATETIME,
    Rank_Type VARCHAR(50),
    Subcategory VARCHAR(50),
    Rank_Value FLOAT
    );
    """

    omega_table = """
    CREATE TABLE OmegaTable(
    ID_INSTRUMENT INT,
    Data DATETIME,
    Omega_Ratio FLOAT
    );
    """

    drop_table_price = """
    DROP TABLE PriceIndex;
    """
    drop_table_r2 = """
    DROP TABLE R2AlphasIndex;
    """
    drop_table_rbsa = """
    DROP TABLE RBSAIndex;
    """
    drop_table_dict = """
    DROP TABLE SociedadeGestora;
    """
    drop_table_fund = """
    DROP TABLE FundIndex;
    """
    execute_query(conn_str,dataRank_table)
    #execute_query(conn_str,BetaIndex_table)


    #Populate the dict
createDataBase()
