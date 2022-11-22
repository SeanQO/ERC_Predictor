import analysis
import pandas as pd

#Load data
###########################################

CSV_PATH = 'src/data/kidney_disease.csv'

df = analysis.load_database(CSV_PATH)

#Data clensing
###########################################

clean_df = pd.DataFrame()

def __data_clensing(df):
  df = analysis.renameCKDColumns(df)
  nan_per_col = analysis.get_nan_per_col(df)

  ACCEPTED_NAN = 10

  df, droped = analysis.removeByNan(ACCEPTED_NAN, nan_per_col, df)
  df = analysis.drop_all_na(df)

  df = analysis.CategoricalVariablesTransformation(df)
  
  df = analysis.foat_int(df)

  clean_df = df

  analysis.print_df(df)
  return clean_df

def get_data_clensing():
  
  clean_df = __data_clensing(df)
  clean_df_head = clean_df.head()
  df_info = __infoOut(clean_df)

  return clean_df_head,df_info

def __infoOut(data,details=False):
  # code from https://stackoverflow.com/questions/64067424/how-to-convert-df-info-into-data-frame-df-info

    dfInfo = data.columns.to_frame(name='Column')
    dfInfo['Non-Null Count'] = data.notna().sum()
    dfInfo['Dtype'] = data.dtypes
    dfInfo.reset_index(drop=True,inplace=True)
    if details:
        rangeIndex = (dfInfo['Non-Null Count'].min(),dfInfo['Non-Null Count'].min())
        totalColumns = dfInfo['Column'].count()
        dtypesCount = dfInfo['Dtype'].value_counts()
        totalMemory = dfInfo.memory_usage().sum()
        return dfInfo, rangeIndex, totalColumns, dtypesCount, totalMemory
    else:
        return dfInfo

#Selecting data and target
###########################################
target = pd.DataFrame()
data = pd.DataFrame()

def __data_and_target():
  clean_df = __data_clensing(df)
  data, target = analysis.getDataAndTarget('classification',clean_df)
  return data, target

def get_data_and_target():
  data, target = __data_and_target()
  data_head = data.head()
  data_info = __infoOut(data)

  target_head = target.head()
  target_info = __infoOut(target)

  return data_head, data_info, target_head, target_info 

#train, test & validation
###########################################
def __train_test_val():
  data, target = __data_and_target()
  X_train, X_test, y_train, y_test, X_val, y_val = analysis.make_train_test_validation(data,target, 0.3, 0.25)
  return X_train, X_test, y_train, y_test, X_val, y_val


def get_train_test_val():
  X_train, X_test, y_train, y_test, X_val, y_val = __train_test_val()
  X_train_info = __infoOut(X_train)
  X_test_info = __infoOut(X_test)

  y_train_info = __infoOut(y_train)
  y_test_info = __infoOut(y_test)

  X_val_info = __infoOut(X_val)
  y_val_info = __infoOut(y_val)
  
  return X_train_info, X_test_info, y_train_info, y_test_info, X_val_info, y_val_info

#Models
###########################################
def __svm():
  X_train, X_test, y_train, y_test, X_val, y_val = __train_test_val()
  train_pred = analysis.tuned_svm(X_train, y_train)
  train_plt = analysis.plotPredVSReqTrain(X_train, y_train, train_pred)
  
  val_predict = analysis.tuned_svm(X_val, y_val) 
  val_plt = analysis.plotPredVSReqValidation(X_train, y_val, val_predict, X_val)
  
  test_predict = analysis.tuned_svm(X_test, y_test)
  test_plt = analysis.plotPredVSReqTest(X_train, y_test, test_predict, X_test)
  
  #train_val_pred = analysis.tuned_svm(X_test, y_test)
  train_val_plt = analysis.plotTrainVSVal(X_train, y_train, X_val, val_predict)

  return train_plt, val_plt, test_plt, train_val_plt

def get_svm():
  train_plt, val_plt, test_plt, train_val_plt = __svm()
  return train_plt, val_plt, test_plt, train_val_plt

def get_svm_tuning():
  X_train, X_test, y_train, _ , _, _ = __train_test_val()
  bestParams, bestEstimator, _ = analysis.svm_tunning(X_train, X_test, y_train)
  return bestEstimator, bestParams
  
def __xgb():
  X_train, X_test, y_train, y_test, X_val, y_val = __train_test_val()
  train_pred = analysis.tuned_xgb(X_train, y_train)
  train_plt = analysis.plotPredVSReqTrain(X_train, y_train, train_pred)
  
  val_predict = analysis.tuned_svm(X_val, y_val) 
  val_plt = analysis.plotPredVSReqValidation(X_train, y_val, val_predict, X_val)
  
  test_predict = analysis.tuned_svm(X_test, y_test)
  test_plt = analysis.plotPredVSReqTest(X_train, y_test, test_predict, X_test)
  
  #train_val_pred = analysis.tuned_svm(X_test, y_test)
  train_val_plt = analysis.plotTrainVSVal(X_train, y_train, X_val, val_predict)

  return train_plt, val_plt, test_plt, train_val_plt
  
def get_xgb():
  train_plt, val_plt, test_plt, train_val_plt = __xgb()
  return train_plt, val_plt, test_plt, train_val_plt 

#Feature Importance
###########################################
def get_importances(X_train, y_train, X_val):
  base_svm = analysis.base_svm_model(X_train, y_train)
  base_xgboost = analysis.base_xgboost_model(X_train, y_train)

  analysis.plot_feature_importance(base_svm, X_train, X_val)
  analysis.plot_feature_importance(base_xgboost, X_train, X_val)