from xgboost import XGBClassifier
import pandas as pd
import numpy as np
import project

def fit():
    X_train, X_test, y_train, y_test, X_val, y_val = project.get_data_splits()
    model = XGBClassifier(reg_lambda = 0.1, 
    reg_alpha = 28, 
    max_depth = 8, 
    learning_rate = 0.000001, 
    gamma = 0.00000001, 
    colsample_bytree = 0.99999999)

    model.fit(X_train, y_train)
    return model

def predict(model, data_to_pred):
    data_to_pred_df = np.array([data_to_pred])
    data_to_pred_df = pd.DataFrame(data_to_pred_df, columns=['age', 'blood_pressure', 'pus_cell_clumps','bacteria','blood_urea', 'serum_creatinine','hypertension','diabetes_mellitus', 'coronary_artery_disease','appetite','peda_edema','anemia'])
    data_to_pred_df = data_to_pred_df.apply(pd.to_numeric)
    prediction = model.predict(data_to_pred_df)
    if(prediction[0] == 1):
        final_pred = "This Patient is prone to have CKD"
    else:
        final_pred = "This Patient is not prone to have CKD"
    
    return final_pred 