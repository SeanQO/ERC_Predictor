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

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)