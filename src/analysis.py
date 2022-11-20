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
