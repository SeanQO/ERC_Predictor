import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from interpret_community.mimic.mimic_explainer import MimicExplainer
from interpret.ext.glassbox import LinearExplainableModel
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, GridSearchCV
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn import metrics
from xgboost import XGBClassifier, plot_tree
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report

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

def foat_int(df):
  
  df['age'] = df['age'].astype(int)
  df['blood_pressure'] = df['blood_pressure'].astype(int)
  df['blood_urea'] = df['blood_urea'].astype(int)
  df['serum_creatinine'] = df['serum_creatinine'].astype(int)

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

#Models
###########################################
#SVM
##################################################
def base_svm_model(X_train, y_train):
  """
    Creates and trains a basic svm model.

    :param X_train: {A dataframe with x train}
    :type X_train: DataFrame.

    :param y_train: {A dataframe with y train}
    :type y_train: DataFrame.

    :return svm_reg: svm regresor model.
    :rtype svm_reg: sklearn.svm._classes.SVC.
    """
  y_train = y_train.values.ravel()
  svm_reg = SVC()
  svm_reg.fit(X_train, y_train)
  return svm_reg

def svm_tuningn(X_train, X_test, y_train):
  """
    tunes the svm hiper parameters(C, gamma and kernel) to find the optimal for the desired prediction.

    :param X_train: {A dataframe with x train}
    :type X_train: DataFrame.

    :param X_test: {A dataframe with x test}
    :type X_test: DataFrame.

    :param y_train: {A dataframe with y train}
    :type y_train: DataFrame.

    :return grid_predictions: ndarray containing prediction resoults.
    :rtype grid_predictions: <class 'numpy.ndarray'>.
    """
  param_grid = {'C': [0.1, 1, 10, 100, 1000], 
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['linear','poly', 'sigmoid']} 
  # 'poly', 'rbf', 'sigmoid', 'precomputed'
  grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)
  grid.fit(X_train, y_train)
  grid_predictions = grid.predict(X_test)

  bestParams = ("Model best params: ", grid.best_params_)
  bestEstimator = ("Model best estimator: ",grid.best_estimator_)

  return (bestParams, bestEstimator, grid_predictions)

def predictAfterTuneSVM(grid, model, X_test, X_val, X_train, y_test):
  """
    returns the predictions after having tuned the model's hyperparameters

    :param grid: {An instance from GridSearchCV}
    :type grid: sklearn.model_selection.GridSearchCV.

    :param model: {An instance from SVC}
    :type model: sklearn.svm._classes.SVC.

    :param X_train: {A dataframe with x train}
    :type X_train: DataFrame.

    :param y_test: {A dataframe with y test}
    :type y_test: DataFrame.

    :param X_test: {A dataframe with x test}
    :type X_test: DataFrame.

    :param X_val: {A dataframe with x val}
    :type X_val: DataFrame.
    """
  grid_predictions = grid.predict(X_test)
  val_predict = model.predict(X_val)
  y_train_pred = model.predict(X_train)

  # print classification report
  print(classification_report(y_test, grid_predictions))

def svm_clasification_report(data, prediction):
  """
    prints a report for svm model according to a reference data and a svm predction

    :param data: {test or validation dataframe split to compare the prediction}
    :type data: DataFrame.

    :param predicion: {A dataframe produced as a resoult of a svm model prediction}
    :type predicion: DataFrame.
    """
  return(classification_report(data, prediction))

#XGBOOST
##################################################
def base_xgboost_model(X_train, y_train, X_test):
  """
    Creates and trains a basic xgboost model.

    :param X_train: {A dataframe with x train}
    :type X_train: DataFrame.

    :param y_train: {A dataframe with y train}
    :type y_train: DataFrame.

    :param X_test: {A dataframe with x test}
    :type X_test: DataFrame.

    :return predicion: A dataframe produced as a resoult of a svm model prediction
    :rtype predicion: DataFrame.
    """
  xgb_Classifier = xgb.XGBClassifier()
  xgb_Classifier.fit(X_train, y_train)
  y_pred = xgb_Classifier.predict(X_test)
  predictions = [round(value) for value in y_pred]
  return predictions

def xgboost_tunning(model, X_train, y_train):
  """
    tunes the hiperparameters of a given xgboost model.

    :param model: {An instance from XGBClassifier}
    :type model: xgboost.sklearn.XGBClassifier.

    :param X_train: {A dataframe with x train}
    :type X_train: DataFrame.

    :param y_train: {A dataframe with y train}
    :type y_train: DataFrame.
    """
  # Define the search space
  param_grid = { 
    # Learning rate shrinks the weights to make the boosting process more conservative
    "learning_rate": [0.0001,0.001, 0.01, 0.1, 1] ,
    # Maximum depth of the tree, increasing it increases the model complexity.
    "max_depth": range(3,21,3),
    # Gamma specifies the minimum loss reduction required to make a split.
    "gamma": [i/10.0 for i in range(0,5)],
    # Percentage of columns to be randomly samples for each tree.
    "colsample_bytree": [i/10.0 for i in range(3,10)],
    # reg_alpha provides l1 regularization to the weight, higher values result in more conservative models
    "reg_alpha": [1e-5, 1e-2, 0.1, 1, 10, 100],
    # reg_lambda provides l2 regularization to the weight, higher values result in more conservative models
    "reg_lambda": [1e-5, 1e-2, 0.1, 1, 10, 100]}
  # Set up score
  scoring = ['recall']
  # Set up the k-fold cross-validation
  kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)

  # Define random search
  random_search = RandomizedSearchCV(estimator=model, 
                            param_distributions=param_grid, 
                            n_iter=48,
                            scoring=scoring, 
                            refit='recall', 
                            n_jobs=-1, 
                            cv=kfold, 
                            verbose=0)
  # Fit grid search
  random_result = random_search.fit(X_train, y_train)
  # Print grid search summary
  random_result
  # Print the best score and the corresponding hyperparameters
  bestScore = (f'The best score is {random_result.best_score_:.4f}')
  bestStandardDeviation = ('The best score standard deviation is', round(random_result.cv_results_['std_test_recall'][random_result.best_index_], 4))
  bestHyperparameters = (f'The best hyperparameters are {random_result.best_params_}')

  return(bestScore, bestStandardDeviation, bestHyperparameters)

def predictAfterTuneXGB(X_train, y_train, X_test, X_val):
  """
    returns the predictions after having tuned the model's hyperparameters

    :param X_train: {A dataframe with x train}
    :type X_train: DataFrame.

    :param y_train: {A dataframe with y train}
    :type y_train: DataFrame.

    :param X_test: {A dataframe with x test}
    :type X_test: DataFrame.

    :param X_val: {A dataframe with x val}
    :type X_val: DataFrame.
    """
  model = XGBClassifier(reg_lambda = 0.1, 
    reg_alpha = 28, 
    max_depth = 8, 
    learning_rate = 0.000001, 
    gamma = 0.00000001, 
    colsample_bytree = 0.99999999)

  model.fit(X_train, y_train)

  y_pred = model.predict(X_test)
  predictions = [round(value) for value in y_pred]
  val_predict = model.predict(X_val)
  y_train_pred = model.predict(X_train)

  return(predictions)
  

def xgboost_clasification_report(data, prediction):
  """
    prints a report for xgboost model according to a reference data and a xgboost predction

    :param data: {test or validation dataframe split to compare the prediction}
    :type data: DataFrame.

    :param predicion: {A dataframe produced as a resoult of a svm model prediction}
    :type predicion: DataFrame.
    """
  return(classification_report(data, prediction))

#Feature Importance
###########################################
def plot_feature_importance(model, X_train, X_val):
  """
    graphs the given model importances based on the data.

    :param X_train: {A dataframe with x train}
    :type X_train: DataFrame.

    :param y_train: {A dataframe with y train}
    :type y_train: DataFrame.

    :param model: {sklearnModel}
    :type model: sklearn.svm._classes.SVC or xgboost.sklearn.XGBClassifier.
    """
  explainer = MimicExplainer(model, X_train, LinearExplainableModel)
  global_explanation = explainer.explain_global(X_val)

  names = np.array(list(global_explanation.get_feature_importance_dict().keys()))
  values = np.array(list(global_explanation.get_feature_importance_dict().values()))

  model_type = str(type(model))
  target = "SVM" if  model_type == "<class 'sklearn.svm._classes.SVC'>" else "XGBoost"

  plt.title('{}  Feature importance'.format(target))
  idx = values.argsort()
  plt.barh(names[idx], values[idx])
  plt.show()

#Accuracy plots
###########################################

def tuned_svm(X_train, y_train):
  """
    Creates and trains a tuned svm model.

    :param X_train: {A dataframe with x train}
    :type X_train: DataFrame.

    :param y_train: {A dataframe with y train}
    :type y_train: DataFrame.

    :return svm_reg: svm regresor model.
    :rtype svm_reg: sklearn.svm._classes.SVC.
    """
  y_train = y_train.values.ravel()
  svm_reg = SVC(kernel="linear",)
  svm_reg.fit(X_train, y_train)
  prediction = svm_reg.predict(X_train)
  return prediction

def tuned_xgb(X_train, y_train, X_test):
  """
    Creates and trains a tuned svm model.

    :param X_train: {A dataframe with x train}
    :type X_train: DataFrame.

    :param y_train: {A dataframe with y train}
    :type y_train: DataFrame.

    :return svm_reg: svm regresor model.
    :rtype svm_reg: sklearn.svm._classes.SVC.
    """
  model = XGBClassifier(reg_lambda = 0.1, 
    reg_alpha = 28, 
    max_depth = 8, 
    learning_rate = 0.000001, 
    gamma = 0.00000001, 
    colsample_bytree = 0.99999999)

  model.fit(X_train, y_train)

  y_pred = model.predict(X_test)
  return y_pred

def plotPredVSReqTrain(X_train, y_train, y_train_pred):
  """
    graphs the accuracy between train set and predicted set.

    :param X_train: {A dataframe with x train}
    :type X_train: DataFrame.

    :param y_train: {A dataframe with y train}
    :type y_train: DataFrame.

    :param y_train_pred: {A dataframe with y train predicted}
    :type y_train_pred: DataFrame.
    """
  y_limits=(-3,3)

  rango_de_salida_de_las_variables_escaladas = (-1,1)
  scaler = MinMaxScaler(feature_range=rango_de_salida_de_las_variables_escaladas)

  scaler.fit(X_train)

  x_train_scaled = scaler.transform(X_train)
  slbls=["y: expected return", "", "", "", "", "", "","", "", "", "", ""]
  plbls=["ypred: predicted return", "", "", "", "", "", "","", "", "", "", ""]

  plt.figure(figsize=(10,7))
  plt.plot(x_train_scaled,y_train,'b.',marker='o', label = slbls)
  plt.plot(x_train_scaled,y_train_pred,'r.',marker='o', label = plbls) 
  plt.xlabel('x escalado.')
  plt.ylabel('Valor de salida.')
  plt.title('Salida deseada y predicción en el conjunto de entrenamiento')
  plt.ylim(y_limits)
  plt.legend()
  fig1 = plt.gcf()
  fig1.savefig("../ERC_Predictor/src/static/plotPredVSReqTrain.jpg", dpi=100)
  #plt.show()
  return "plotPredVSReqTrain.jpg"

def plotPredVSReqValidation(X_train, y_val, val_predict, X_val):
  """
    graphs the accuracy between validation set and predicted set.

    :param X_train: {A dataframe with x train}
    :type X_train: DataFrame.

    :param y_val: {A dataframe with y valiation set}
    :type y_val: DataFrame.

    :param val_predict: {A dataframe with y valiation set predicted}
    :type val_predict: DataFrame.

    :param x_val: {A dataframe with x valiation set}
    :type x_val: DataFrame.
    """
  y_limits=(-3,3)

  rango_de_salida_de_las_variables_escaladas = (-1,1)
  scaler = MinMaxScaler(feature_range=rango_de_salida_de_las_variables_escaladas) 

  scaler.fit(X_train)

  x_val_scaled   = scaler.transform(X_val)

  slbls=["y: expected return", "", "", "", "", "", "","", "", "", "", ""]
  plbls=["ypred: predicted return", "", "", "", "", "", "","", "", "", "", ""]

  plt.figure(figsize=(10,7))
  plt.plot(x_val_scaled,y_val,'b.',marker='o', label= slbls)
  plt.plot(x_val_scaled,val_predict,'r.',marker='o', label=plbls) 
  plt.xlabel('x escalado.')
  plt.ylabel('Valor de salida.')
  plt.title('Salida deseada y predicción en el conjunto de validación')
  plt.ylim(y_limits)
  plt.legend()
  fig1 = plt.gcf()
  fig1.savefig("../ERC_Predictor/src/static/plotPredVSReqValidation.jpg", dpi=100)
  #plt.show()
  return "plotPredVSReqValidation.jpg"

def plotPredVSReqTest(X_train, y_test, predictions, X_test):
  """
    graphs the accuracy between test set and predicted set.

    :param X_train: {A dataframe with x train}
    :type X_train: DataFrame.

    :param y_test: {A dataframe with y test set}
    :type y_test: DataFrame.

    :param predictions: {A dataframe with y set predicted}
    :type predictions: DataFrame.

    :param x_test: {A dataframe with x test set}
    :type x_test: DataFrame.
    """
  y_limits=(-3,3)

  rango_de_salida_de_las_variables_escaladas = (-1,1)  #Tupla con el siguiente formato: (mínimo deseado, máximo deseado).
  scaler = MinMaxScaler(feature_range=rango_de_salida_de_las_variables_escaladas)  #Instanciamos el objeto para escalar los datos. 

  scaler.fit(X_train) #Ajustamos los datos de entrenamiento.

  x_test_scaled  = scaler.transform(X_test)   #Transformamos los datos de pruebas
  slbls=["y: expected return", "", "", "", "", "", "","", "", "", "", ""]
  plbls=["ypred: predicted return", "", "", "", "", "", "","", "", "", "", ""]

  plt.figure(figsize=(10,7))
  plt.plot(x_test_scaled,y_test,'b.',marker='o', label=slbls)
  plt.plot(x_test_scaled,predictions,'r.',marker='o', label=plbls) 
  plt.xlabel('x escalado.')
  plt.ylabel('Valor de salida.')
  plt.title('Salida deseada y predicción en el conjunto de pruebas')
  plt.ylim(y_limits)
  plt.legend()
  fig1 = plt.gcf()
  fig1.savefig("../ERC_Predictor/src/static/plotPredVSReqTest.jpg", dpi=100)
  #plt.show()
  return "plotPredVSReqTest.jpg"

def plotTrainVSVal(X_train, y_train, X_val, val_predict):
  """
    graphs the accuracy between test set and predicted set.

    :param X_train: {A dataframe with x train}
    :type X_train: DataFrame.

    :param y_train: {A dataframe with y train set}
    :type y_test: DataFrame.

    :param x_val: {A dataframe with x validation set}
    :type x_val: DataFrame.

    :param val_predict: {A dataframe with y validation set predicted}
    :type val_predict: DataFrame.
    """
  y_limits=(-3,3)

  rango_de_salida_de_las_variables_escaladas = (-1,1)  #Tupla con el siguiente formato: (mínimo deseado, máximo deseado).
  scaler = MinMaxScaler(feature_range=rango_de_salida_de_las_variables_escaladas)  #Instanciamos el objeto para escalar los datos. 

  scaler.fit(X_train) #Ajustamos los datos de entrenamiento.

  x_train_scaled = scaler.transform(X_train)  #Transformamos los datos de entrenamiento.
  x_val_scaled   = scaler.transform(X_val)    #Transformamos los datos de validación.
  slbls=["y: expected return", "", "", "", "", "", "","", "", "", "", ""]
  plbls=["ypred: predicted return", "", "", "", "", "", "","", "", "", "", ""]

  plt.figure(figsize=(10,7))
  plt.plot(x_train_scaled,y_train,'b.',marker='o', label=slbls)
  plt.plot(x_val_scaled,val_predict,'r.',marker='o', label=plbls) 
  plt.xlabel('x train escalado.')
  plt.ylabel('X Validation escalado.')
  plt.title('conjunto de training vs conjunto de validación')
  plt.ylim(y_limits)
  plt.legend()
  fig1 = plt.gcf()
  fig1.savefig("../ERC_Predictor/src/static/plotTrainVSVal.jpg", dpi=100)
  #plt.show()
  return "plotTrainVSVal.jpg"