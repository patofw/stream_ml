import pickle

import pandas as pd
from flask import Flask

from config import Config


# Set some paths


MODEL_DIR = Config.MODEL_PATH
# For this example, we load simulated data
# as it was a user interaction.
# This ideally should come from the app as a json input
FAKE_DATA_INPUT = Config.FAKE_USER_TST

# load our model 
with open(MODEL_DIR, "rb") as f:
    online_model = pickle.load(f)
# Creation of the Flask app
app = Flask(__name__)

# API 1
# Flask route so that we can serve HTTP traffic on that route
# This would mimic getting data from the user


@app.route('/line/<Line>')
# Get data from json and return the requested row defined by the variable Line
def line(Line):

    file_data = pd.read_pickle(FAKE_DATA_INPUT)
    # We can then find the data for the requested row and send it back as json
    return file_data.iloc[int(Line)].to_json()


# API 2
# Flask route so that we can serve HTTP traffic on that route
# This is where we are going to make predictions and retrain the model.
@app.route('/prediction/<int:Line>', methods=['POST', 'GET'])
def prediction(Line):
    # load data
    file_data = pd.read_pickle(FAKE_DATA_INPUT)
    # load model and make sure we are constantly
    # updating it
    global online_model  # make it accesible by other methods
    # predict
    pred, after_roc_score = online_model.new_entry(
       file_data.iloc[int(Line)],
       plot_metrics=False
    )

    return {
        "PREDICTION SCORE": pred,
        "ROC SCORE AFTER PREDICTION": str(after_roc_score)
    }

# API 3
# Flask route so that we can serve HTTP traffic on that route


@app.route('/score', methods=['POST', 'GET'])
# Return classification score
def score():

    return {
        'Score Model': str(online_model._metric),
        'Model Confusion matrix': str(online_model._confusion_matrix)
    }


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5100)
