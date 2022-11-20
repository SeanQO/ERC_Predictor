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
  print(df.head())

#Data clensing
###########################################

def renameCKDColumns(df):
  """
    Replaces all column default names for more legible ones.
    New names:'id','age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar', 'red_blood_cells', 'pus_cell','pus_cell_clumps', 'bacteria', 'blood_glucose_random', 'blood_urea', 'serum_creatinine', 'sodium',
              'potassium', 'haemoglobin', 'packed_cell_volume', 'white_blood_cell_count', 'red_blood_cell_count',
              'hypertension', 'diabetes_mellitus', 'coronary_artery_disease', 'appetite', 'peda_edema',
              'anemia', 'classification'.

    :param df: {pandas data frame}.
    :type df: DataFrame.
    :return: A dataframe with new column names.
    :rtype: DataFrame.

    """
  df.columns = ['id','age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar', 'red_blood_cells', 'pus_cell',
              'pus_cell_clumps', 'bacteria', 'blood_glucose_random', 'blood_urea', 'serum_creatinine', 'sodium',
              'potassium', 'haemoglobin', 'packed_cell_volume', 'white_blood_cell_count', 'red_blood_cell_count',
              'hypertension', 'diabetes_mellitus', 'coronary_artery_disease', 'appetite', 'peda_edema',
              'anemia', 'classification']
  return df

def get_nan_per_col(df):
  """
    Iterates trough every column and caculates the percentage of nan values present in each column.

    :param df: {pandas data frame}.
    :type df: DataFrame.
    :return: a series of all the nan percentages asociated to the column name.
    :rtype: series.

    """
  NAN_percentage = ((df.isna().sum()  * 100) / df.shape[0]).sort_values(ascending=True)
  return NAN_percentage

def removeByNan(accepted_nan_percentage, nan_percentaje_series,df):
  """
    Takes a series that contains all the columns name with the respective nan percentaje, 
    then drops the columns that had nan content above the diven max nan limit.

    :param accepted_nan_percentage: {positive number that represent the limit of nan % acceptance}.
    :param nan_percentaje_series: {series that contains all the columns name with the respective nan percentaje}.
    :param df: {pandas data frame}.
    :type accepted_nan_percentage: float.
    :type nan_percentaje_series: series.
    :type df: DataFrame.
    :return df: A dataframe without the columns that didint comply with the limit.
    :rtype df: DataFrame.
    :return columns_droped: list of the name of the droped columns and its nan percentage.
    :rtype columns_droped: list[str,float].

    """
  columns_droped = []
  for items in nan_percentaje_series.iteritems():
      if items[1] > accepted_nan_percentage: 
        df = df.drop([items[0]],axis=1)
        columns_droped.append(items)
  
  return df,columns_droped

def drop_all_na(df):
  """
    Drops all rows that contains nan values 

    :param df: {pandas data frame}.
    :type df: DataFrame.
    :return: A dataframe without nan values.
    :rtype: DataFrame.

    """
  df = df.dropna() 
  return df

def CategoricalVariablesTransformation(df):
  """
    Transforms categorical variables in the dataframe by replacing each unique value in a categorical column for an integer as a representation.

    :param df: {pandas data frame}.
    :type df: DataFrame.
    :return: A dataframe with the categorical columns transformed into integers.
    :rtype: DataFrame.

    """
  df = __yes_and_no_transform(df)
  
  df = df.replace('ckd\t','ckd')
  df = __replace('classification',df)

  df = __replace('pus_cell_clumps',df)
  df = __replace('bacteria',df)

  df = df.replace('good',1)
  df = df.replace('poor',0)
  
  return df

def __yes_and_no_transform(df):
  """
    Assigns and replace all yes/no columns for 1/0 integers.

    :param df: {pandas data frame}.
    :type df: DataFrame.
    :return: A dataframe with the yes/no format columns transformed into integers.
    :rtype: DataFrame.

    """
  df = df.replace('yes',1)
  df = df.replace('no',0)

  df = df.replace('\tno', 0)
  df = df.replace('\tyes', 1)

  df = df.replace('no',0)

  return df

def __replace(col_name,df):
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

#Selecting data and target
###########################################

def getDataAndTarget(target_col,df):
  """
    Splits the data frame into target(column name given) and data(main dataframe without target and id columns)

    :param target_col: {Target column name}.
    :param df: {pandas data frame}.
    :type target_col: str.
    :type df: DataFrame.
    :return data: A dataframe without target and id columns.
    :rtype data: DataFrame.
    :return target: A dataframe with just the target column
    :rtype target: DataFrame.

    """
  data = df.drop([target_col,'id'], axis=1)
  target = df[[target_col]]

  return data,target

#train, test & validation
###########################################
def make_train_test_validation(data,target, testSize, validationSize):
  """
    Splits the data frame into train, test & validation with the specified sizes.

    :param data: {A dataframe without target and id columns}.
    :type data: DataFrame.
    :param target: {A dataframe with just the target column}.
    :type target: DataFrame.
    :return X_train: A dataframe with x train
    :rtype X_train: DataFrame.

    :return X_test: A dataframe with x test
    :rtype X_test: DataFrame.

    :return y_train: A dataframe with y train
    :rtype y_train: DataFrame.

    :return y_test: A dataframe with y test
    :rtype y_test: DataFrame.

    :return X_val: A dataframe with x val
    :rtype X_val: DataFrame.

    :return y_val: A dataframe with y val
    :rtype y_val: DataFrame.

    """
  X_train, X_test, y_train, y_test = train_test_split(data, target,
    test_size=testSize, shuffle = True, random_state = 8)

  X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
    test_size=validationSize, random_state= 8)
  
  return X_train, X_test, y_train, y_test, X_val, y_val

def print_splits_shapes(X_train, X_test, y_train, y_test, X_val, y_val):
  """
    prints train, test & validation final dimentions.

    :param X_train: {A dataframe with x train}
    :type X_train: DataFrame.

    :param X_test: {A dataframe with x test}
    :type X_test: DataFrame.

    :param y_train: {A dataframe with y train}
    :type y_train: DataFrame.

    :param y_test: {A dataframe with y test}
    :type y_test: DataFrame.

    :param X_val: {A dataframe with x val}
    :type X_val: DataFrame.

    :param y_val: {A dataframe with y val}
    :type y_val: DataFrame.

    """
  print("Final Shapes:")
  print("X_train: {}".format(X_train.shape))
  print("X_test: {}".format(X_test.shape))
  print("y_train: {}".format(y_train.shape))
  print("y_test: {}".format(y_test.shape))
  print("X_val: {}".format(X_val.shape))
  print("y val: {}".format(y_val.shape))