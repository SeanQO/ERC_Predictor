import project

#Load data
###########################################

CSV_PATH = 'src/data/kidney_disease.csv'

df = project.load_database(CSV_PATH)

project.print_df(df)

#Data clensing
###########################################

df = project.renameCKDColumns(df)
nan_per_col = project.get_nan_per_col(df)

ACCEPTED_NAN = 10

df, droped = project.removeByNan(ACCEPTED_NAN, nan_per_col, df)
df = project.drop_all_na(df)

df = project.CategoricalVariablesTransformation(df)

project.print_df(df)

#Selecting data and target
###########################################
data, target = project.getDataAndTarget('classification',df)

project.print_df(data)
project.print_df(target)

#train, test & validation
###########################################

X_train, X_test, y_train, y_test, X_val, y_val = project.make_train_test_validation(data,target, 0.3, 0.25)
project.print_splits_shapes(X_train, X_test, y_train, y_test, X_val, y_val)

#Models
###########################################

svm_prediction = project.svm_tunning(X_train, X_test, y_train)
project.svm_clasification_report(y_test, svm_prediction)

xgb_Classifier = project.base_xgboost_model(X_train, y_train)
prediction = project.xgboost_tunning(X_test, xgb_Classifier)
project.xgboost_clasification_report(y_test, prediction)

#Feature Importance
###########################################
base_svm = project.base_svm_model(X_train, y_train)
base_xgboost = project.base_xgboost_model(X_train, y_train)

project.plot_feature_importance(base_svm, X_train, X_val)
project.plot_feature_importance(base_xgboost, X_train, X_val)





