import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})
plt.style.use('ggplot')

# Gives brief description about data
# Arguments:
#   data we want to analyze
#     - data : pd.DataFrame
# Returns:
#     - None
def brief_data_analyse(data: pd.DataFrame):
    print('======= First 5 records =======')
    print(data.head(5))

    print('======= Info =======')
    print(data.info())

    print('======= Dtypes =======')
    print(data.dtypes)

    print('======= Missing values =======')
    print(data.isnull().sum())

    print('======= Null values =======')
    print(data.isna().sum())

    print('======= Data Shape =======')
    print(data.shape)

# Builds graphs on numeric variables of data set
# Arguments:
#   data we want to analyze
#      - data : pd.DataFrame
# Returns:
#      - None
def graph_insight(data : pd.DataFrame, figsize=(16, 16)):
  # printing all data types in data frame
  print(set(data.dtypes.tolist()))

  # getting only numeric columns
  df_num = data.select_dtypes(include = ['float64', 'int64'])

  # plotting histograms on numeric columns
  df_num.hist(figsize=figsize, bins=50, xlabelsize=8, ylabelsize=9)


# Drops duplicates from data and print information about dropping
# Arguments:
#   data we want to drop duplicates from and subset of columns
#   that will be checked
#      - data : pd.DataFrame, subset : list
# Returns:
#      - None
def drop_duplicates(data: pd.DataFrame, subset: list):
    # printing shape of data before dropping values
    print('Shape before drop: ', data.shape)

    # getting number of records before dropping
    before = data.shape[0]

    # dropping duplicates
    data.drop_duplicates(subset, keep='first', inplace=True)

    # reseting index
    data.reset_index(drop=True, inplace=True)

    # printing data shape after dropping
    print("Shape after drop: ", data.shape)

    # getting number of records after dropping
    after = data.shape[0]

    # printing number of duplicated dropped
    print('Number of duplicates: ', before - after)

# Checks for unreal data
# Arguments:
#   data we want to check
#      - data : pd.Series
# Returns:
#     - None
def unreal_data_check(data : pd.Series):
  print('Min value: ', data.min())
  print('Max value: ', data.max())
  print('Average value: ', data.mean())
  print('Median value: ', data.median())


# Performing stationary test
# Arguments:
#     data we want to analyze
#       - time_series : pd.DataFrame
#       -
def stationarity_test(time_series : pd.DataFrame):
    # Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')

    df_test = adfuller(time_series , autolag='AIC')
    df_output = pd.Series(df_test[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])

    for key, value in df_test[4].items():
        df_output['Critical Value (%s)' % key] = value

    print(df_output)

# Getting difference at interval
# Arguments:
#   data set we want to transform and interval
#   of calculating difference
#     - data_set : pd.DataFrame, interval : int = 1
def difference(data_set : pd.DataFrame, interval : int = 1):
    diff = []

    for i in range(interval, len(data_set)):
        value = data_set[i] - data_set[i - interval]
        diff.append(value)

    return pd.Series(diff)

# Getting back to original data
# Arguments:
#    last object and value of current object
#       - last_ob, value
# Returns:
#   inverse difference
def inverse_difference(last_ob, value):
    return value + last_ob