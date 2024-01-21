import pickle

import pandas as pd
from flask import Flask, render_template, request

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
    # First lets load data and set defaults
    file_data = pd.read_pickle(FAKE_DATA_INPUT)
    post_data = file_data.iloc[int(Line)]
    selected_option = None
    user_input = 0  # by default the user does not accept the post
    prediction_result = {}
    new_roc = {}

    # now let's make a prediction
    # load model and make sure we are constantly
    # updating it by makint it a global variable
    global online_model  # make it accesible by other methods
    # make initial prediction
    pred = online_model._model.predict_proba_one(post_data)
    # get the output of the model.
    prediction_result = {
        "PREDICTION SCORE": pred,
    }

    if request.method == 'POST':
        selected_option = request.form.get('options')
        # You can now use the selected_option for the model
        # we pass the selection of the user to the model for learning
        if selected_option == "accepted":
            user_input = 1
        online_model._return_fake_user_input = user_input

        # retrain the model with user input
        _, after_roc_score = online_model.new_entry(
            post_data,
            plot_metrics=False
        )

        new_roc = {
            "ROC SCORE AFTER PREDICTION": str(after_roc_score)
        }
    return render_template(
        'prediction.html',
        selected_option=selected_option,
        pred_result=prediction_result,
        post_id=str(post_data.tweet_id),
        new_roc=new_roc
    )

# API 3
# Flask route so that we can serve HTTP traffic on that route


@app.route('/score', methods=['POST', 'GET'])
# Return classification score
def score():

    return {
        'ROC Score Model': str(online_model._metric),
    }


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5100)
