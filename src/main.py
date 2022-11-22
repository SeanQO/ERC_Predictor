from flask import Flask
from flask import request,url_for
import project
import pandas as pd

app = Flask(__name__)

@app.route("/")
def index():
    return(
        f"""<div>
                <ul>
                    <li><a href="{url_for('model')}">Model</a></li>
                    <li><a href="{url_for('data_clensing')}">Data clensing</a></li>
                    <li><a href="{url_for('data_target')}">Data and target split</a></li>
                    <li><a href="{url_for('train_test_val')}">Train,test and validation splits split</a></li>
                    <li><a href="{url_for('svm')}">SVM analysis</a></li>
                    <li><a href="{url_for('xgb')}">XGB analysis</a></li>
                </ul>
            </div>
            """        
    )

@app.route("/model")
def model():
    target = request.args.get("target", "")
    return(
        """<div>
                <form action="" method="get">
                    <input type="text" name="target">
                    <input type="submit" value="predict">
                </form>
            </div>
            """
              + target
    )

@app.route("/analysis/data_clensing")
def data_clensing():
    clean_df, df_info = project.get_data_clensing()
    clean_df_html = pd.DataFrame(data=clean_df).to_html()
    df_info = pd.DataFrame(data=df_info).to_html()

    return f'<h1>Clean Data frame:</h1>{clean_df_html}{df_info}'

@app.route("/analysis/data_target")
def data_target():
    data_head, data_info, target_head, target_info = project.get_data_and_target()
    data_head_html = pd.DataFrame(data=data_head).to_html()
    data_info_html = pd.DataFrame(data=data_info).to_html()
    target_head_html = pd.DataFrame(data=target_head).to_html()
    target_info_html = pd.DataFrame(data=target_info).to_html()

    return f"""<div>
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