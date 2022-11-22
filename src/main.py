from flask import Flask
from flask import request,url_for
import project
import final_model
import pandas as pd

app = Flask(__name__)

@app.route("/")
def index():
    return(
        f"""<div>
                <head>
                    <h1 style="text-align:center;font-size:300%;">ERC Predictor</h1>
                </head>
                <body style="background-color:lightgrey; text-align:center;">
                    <ul>
                        <br>
                        <a href="{url_for('model')}"><input type=button value="Model" style="background:darkcyan;color:white;"></a>
                        </br>
                        <br>
                        <a href="{url_for('data_clensing')}"><input type=button value="Data clensing" style="background:darkcyan;color:white;"></a>
                        </br>
                        <br>
                        <a href="{url_for('data_target')}"><input type=button value="Data and target split" style="background:darkcyan;color:white;"></a>
                        </br>
                        <br>
                        <a href="{url_for('train_test_val')}"><input type=button value="Train,test and validation splits split" style="background:darkcyan;color:white;"></a>
                        </br>
                        <br>
                        <a href="{url_for('svm')}"><input type=button value="SVM analysis" style="background:darkcyan;color:white;"></a>
                        </br>
                        <br>
                        <a href="{url_for('xgb')}"><input type=button value="XGB analysis" style="background:darkcyan;color:white;"></a>
                        </br>
                    </ul>
                </body>
            </div>
            """        
    )

@app.route("/model")
def model():
    filled = False
    n = 0
    pred = ""
    
    age = request.args.get("age", "")
    blood_pressure = request.args.get("blood_pressure", "")
    pus_cell_clumps = request.args.get("pus_cell_clumps", "")
    bacteria = request.args.get("bacteria", "")
    blood_urea = request.args.get("blood_urea", "")
    serum_creatinine = request.args.get("serum_creatinine", "")
    hypertension = request.args.get("hypertension", "")
    diabetes_mellitus = request.args.get("diabetes_mellitus", "")
    coronary_artery_disease = request.args.get("coronary_artery_disease", "")
    appetite = request.args.get("appetite", "")
    peda_edema = request.args.get("peda_edema", "")
    anemia = request.args.get("anemia", "")

    data_to_pred = [age,blood_pressure,pus_cell_clumps,bacteria,blood_urea, serum_creatinine,hypertension,diabetes_mellitus,coronary_artery_disease,appetite, peda_edema, anemia]


    for item in data_to_pred:
        if(item == ""):
            n = n+1
    
    if(n == 0):
        filled = True

    n = 0

    if (filled):
        age = int(request.args.get("age", ""))
        blood_pressure = int(request.args.get("blood_pressure", ""))
        pus_cell_clumps = int(request.args.get("pus_cell_clumps", ""))
        bacteria = int(request.args.get("bacteria", ""))
        blood_urea = int(request.args.get("blood_urea", ""))
        serum_creatinine = int(request.args.get("serum_creatinine", ""))
        hypertension = int(request.args.get("hypertension", ""))
        diabetes_mellitus = int(request.args.get("diabetes_mellitus", ""))
        coronary_artery_disease = int(request.args.get("coronary_artery_disease", ""))
        appetite = int(request.args.get("appetite", ""))
        peda_edema = int(request.args.get("peda_edema", ""))
        anemia = int(request.args.get("anemia", ""))
        model = final_model.fit()
        pred = final_model.predict(model,data_to_pred)

    return(
        f"""<div>
                <ul>
                    <li><a href="{url_for('index')}">Go back</a></li>
                </ul>
                <form action="" method="get">
                    <ul>
                        <input type="text" name="age">
                        <input type="text" name="blood_pressure">
                        <input type="text" name="pus_cell_clumps">
                        <input type="text" name="bacteria">
                        <input type="text" name="blood_urea">
                        <input type="text" name="serum_creatinine">
                        <input type="text" name="hypertension">
                        <input type="text" name="diabetes_mellitus">
                        <input type="text" name="coronary_artery_disease">
                        <input type="text" name="appetite">
                        <input type="text" name="peda_edema">
                        <input type="text" name="anemia">
                        <input type="submit" value="predict">
                    </ul>   
                </form>
                <div>
                    <h1>{pred}</h1>
                </div>
            </div>
            """
    )

@app.route("/analysis/data_clensing")
def data_clensing():
    clean_df, df_info = project.get_data_clensing()
    clean_df_html = pd.DataFrame(data=clean_df).to_html()
    df_info = pd.DataFrame(data=df_info).to_html()

    return f"""<div>
                <ul>
                    <li><a href="{url_for('index')}">Go back</a></li>
                </ul>
                <h1>Clean Data frame:</h1>
                {clean_df_html}
                {df_info}
            </div>
            """ 

@app.route("/analysis/data_target")
def data_target():
    data_head, data_info, target_head, target_info = project.get_data_and_target()
    data_head_html = pd.DataFrame(data=data_head).to_html()
    data_info_html = pd.DataFrame(data=data_info).to_html()
    target_head_html = pd.DataFrame(data=target_head).to_html()
    target_info_html = pd.DataFrame(data=target_info).to_html()

    return f"""<div>
                    <ul>
                        <li><a href="{url_for('index')}">Go back</a></li>
                    </ul>
                    <h1>Data and target split:</h1>
                    <br/>
                    <h1>Data split:</h1>
                    {data_head_html}{data_info_html}
                    <h1>Target split:</h1>
                    {target_head_html}{target_info_html}
                </div> 
                """

@app.route("/analysis/train_test_val")
def train_test_val():
    X_train_info, X_test_info, y_train_info, y_test_info, X_val_info, y_val_info = project.get_train_test_val()
    X_train_info_html = pd.DataFrame(data=X_train_info).to_html()
    X_test_info_html = pd.DataFrame(data=X_test_info).to_html()
    y_train_info_html = pd.DataFrame(data=y_train_info).to_html()
    y_test_info_html = pd.DataFrame(data=y_test_info).to_html()
    X_val_info_html = pd.DataFrame(data=X_val_info).to_html()
    y_val_info_html = pd.DataFrame(data=y_val_info).to_html()

    return f"""<div>
                    <ul>
                        <li><a href="{url_for('index')}">Go back</a></li>
                    </ul>
                    <h1>Train test validation:</h1>
                    <br/>
                    <h1>X_train:</h1>
                    {X_train_info_html}
                    <h1>X_test:</h1>
                    {X_test_info_html}
                    <h1>y_train:</h1>
                    {y_train_info_html}
                    <h1>y_test:</h1>
                    {y_test_info_html}
                    <h1>X_val:</h1>
                    {X_val_info_html}
                    <h1>y_val:</h1>
                    {y_val_info_html}
                </div> 
                """

@app.route("/analysis/svm")
def svm():
    train_plt, val_plt, test_plt, train_val_plt = project.get_svm()
    return f"""<div>
                    <ul>
                        <li><a href="{url_for('tunning')}">Tunning</a></li>
                        <li><a href="{url_for('index')}">Go back</a></li>
                    </ul>
                    <h1>SVM:</h1>
                     <h2>Pred VS Train:</h2>
                    <img src="/static/{train_plt}" />
                    <h2>Pred VS Val:</h2>
                    <img src="/static/{val_plt}" />
                    <h2>Pred VS Train:</h2>
                    <img src="/static/{test_plt}" />
                    <h2>Train VS Val:</h2>
                    <img src="/static/{train_val_plt}" />
                </div> 
                """
@app.route("/tunning")
def tunning():
    bestEstimator, bestParams = project.get_svm_tuning()
    return f"""<div>
                    <ul>
                        <li><a href="{url_for('svm')}">Go back</a></li>
                    </ul>
                    <h1>Tunning for:</h1>
                     <h2>Best parameters: {bestParams}</h2>
                     <h2>Best estimator: {bestEstimator}<h2/>
                </div> 
                """
    

@app.route("/analysis/xgb")
def xgb():
    train_plt, val_plt, test_plt, train_val_plt = project.get_xgb()
    return f"""<div>
                    <h1>XGB:</h1>
                     <h2>Pred VS Train:</h2>
                    <img src="/static/{train_plt}" />
                    <h2>Pred VS Val:</h2>
                    <img src="/static/{val_plt}" />
                    <h2>Pred VS Train:</h2>
                    <img src="/static/{test_plt}" />
                    <h2>Train VS Val:</h2>
                    <img src="/static/{train_val_plt}" />
                </div> 
                """

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)