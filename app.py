from flask import Flask, request, jsonify, render_template

import pandas as pd
import numpy as np

import pickle


app = Flask(__name__)
# Load model
model = pickle.load(open("templates/model.pkl", "rb"))
# Load vectorizer
vectorizer = pickle.load(open('templates/vectorizer.pkl','rb'))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/index.html")
def go_home():
    return render_template("index.html")

@app.route("/prediction.html")
def go_to_prediction():
    return render_template("prediction.html")


@app.route("/prediction", methods = ["POST", "GET"])
def predict(model = model,
            vectorizer = vectorizer):

    text = request.form["lyrics"]

    # String text
    text = str(text)
    # Lowercase text
    text = text.lower()

    # Vectorize new lyrics
    lyric_vec = vectorizer.transform([text])

    # Predict Genres
    pred = model.predict(lyric_vec)

    # Prediction Output
    print(pred)

    # Show results on HTML
    return render_template("prediciton.html", prediction_string="Predictions: ", )

if __name__ == "__main__":
    app.run()