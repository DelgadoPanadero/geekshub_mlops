import csv
import pickle
from io import StringIO

import flask
import joblib
import pandas as pd


# cargamos modelo
model = joblib.load('/opt/ml/model/model.joblib')

app = flask.Flask(__name__)

@app.route("/ping", methods=["GET"])
def ping():
    return flask.Response(response="\n", status=200,
        mimetype="application/json")


@app.route("/invocations", methods=["POST"])
def predict():

    # Convert from CSV to pandas
    if flask.request.content_type == "text/csv":
        data = flask.request.data.decode("utf-8")
        s = StringIO(data)
        X = pd.read_csv(s).values[:,:-1]

        # Do the prediction
        predictions = pd.DataFrame(
            model.predict_proba(X), columns=model.classes_)

        # Convert from numpy back to CSV
        out = StringIO()
        predictions.to_csv(out, header=False, index=False)
        result = out.getvalue()

        return flask.Response(
            response=result,
            status=200,
            mimetype="text/csv")


app.run(host="0.0.0.0", port=8080, debug=True)
