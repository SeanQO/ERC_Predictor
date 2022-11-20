import analysis
import pandas as pd

#Load data
###########################################

CSV_PATH = 'src/data/kidney_disease.csv'

df = analysis.load_database(CSV_PATH)

analysis.print_df(df)

#Data clensing
###########################################

clean_df = pd.DataFrame()

def __data_clensing(df):
  df = analysis.renameCKDColumns(df)
  nan_per_col = analysis.get_nan_per_col(df)

  ACCEPTED_NAN = 10

  df, droped = analysis.removeByNan(ACCEPTED_NAN, nan_per_col, df)
  df = analysis.drop_all_na(df)

  clean_df = analysis.CategoricalVariablesTransformation(df)

  analysis.print_df(df)
  return clean_df

def get_data_clensing():
  clean_df = __data_clensing(df)
  clean_df_head = clean_df.head()
  df_info = infoOut(clean_df)

  return clean_df_head,df_info

def infoOut(data,details=False):
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

def _temp():
  #Selecting data and target
  ###########################################
  data, target = analysis.getDataAndTarget('classification',df)

  analysis.print_df(data)
  analysis.print_df(target)

  #train, test & validation
  ###########################################

  X_train, X_test, y_train, y_test, X_val, y_val = analysis.make_train_test_validation(data,target, 0.3, 0.25)
  analysis.print_splits_shapes(X_train, X_test, y_train, y_test, X_val, y_val)

  #Models
  ###########################################

  svm_prediction = analysis.svm_tunning(X_train, X_test, y_train)
  analysis.svm_clasification_report(y_test, svm_prediction)

  xgb_Classifier = analysis.base_xgboost_model(X_train, y_train)
  prediction = analysis.xgboost_tunning(X_test, xgb_Classifier)
  analysis.xgboost_clasification_report(y_test, prediction)

  #Feature Importance
  ###########################################
  base_svm = analysis.base_svm_model(X_train, y_train)
  base_xgboost = analysis.base_xgboost_model(X_train, y_train)

  analysis.plot_feature_importance(base_svm, X_train, X_val)
  analysis.plot_feature_importance(base_xgboost, X_train, X_val)

