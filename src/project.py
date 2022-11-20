import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from interpret_community.mimic.mimic_explainer import MimicExplainer
from interpret.ext.glassbox import LinearExplainableModel
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn import metrics


def load_database(path):
  """
    Loads a csv database from a given path.

    :param path: {path of the csv file}.
    :type path: str.
    :return: A dataframe loaded with the csv information.
    :rtype: DataFrame.

    """
  df = pd.read_csv(path)
  return df

def print_df(df):
  """
    prints the basic information of a given data frame

    :param df: {pandas data frame}.
    :type df: DataFrame

    """
  print(df)
  print(df.info())

def replace(col_name,df):
  """
    Assigns and replace all unique values of a column to a integer.
    For each unique value found a number from 0 is asigned.

    :param col_name: {Target column name}.
    :param df: {pandas data frame}.
    :type col_name: str.
    :type df: DataFrame.
    :return: A dataframe with the target column turn into categoric values.
    :rtype: DataFrame.

    """
  events_lst = df[col_name].unique().tolist()
  print("unique values: ",events_lst)

  print("mapping unique values to int")
  for i in range(len(events_lst)):
    df[col_name] = df[col_name].replace(events_lst[i],i)
    print("{} : {}".format(events_lst[i],i))

  events_lst = df[col_name].unique().tolist()
  print("unique values: ",events_lst)
  
  return df