from flask import Flask, request, jsonify, render_template

import pandas as pd
import numpy as np
from string import punctuation
from nltk.corpus import stopwords
import re
from better_profanity import profanity

import pickle


app = Flask(__name__)
# Load model
model = pickle.load(open('templates/model.pkl', 'rb'))

# Load vectorizer
vectorizer = pickle.load(open('templates/vectorizer.pkl','rb'))

@app.route("/", methods = ["POST", "GET"])
def home():
    return render_template("index.html")

@app.route("/index.html", methods = ["POST", "GET"])
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

    # Remove any possible ads in middle of genius lyrics
    text = re.sub(r"(See[\w\s\W]+?\$[\d,]+You might also like)", "", text)

    # Remove any junk at end of genius lyrics
    text = re.sub(r"([\d]+Embed)","", text)

    # Lowercase text
    text = text.lower()

    # Remove stuff in brackets
    text = re.sub("[\(\[].*?[\)\]]", "", text)

    # Remove profanity
    text = profanity.censor(text)

    # Tokenize
    tokens = text.strip().replace("\n","\s").split()

    # Remove Punctuation
    punct = set(punctuation)
    punct_no_apo = punct - {"'", "*"}
    punct_prof = punct - {"*"}
    tokens = ["".join(ch for ch in word if ch not in punct_no_apo) for word in tokens]

    # Remove Stop Words
    # Stopwords
    sw = set(stopwords.words("english"))
    # Removing words that were common in our word clouds
    stop_add = ['like', 'yeah', 'oh', 'i\'m', 'i\'ve', 'i\'ll', 'can\'t', 'cause']
    sw.update(stop_add)
    tokens = [t for t in tokens if t not in sw]

    # Remove more punctuations
    tokens = ["".join(ch for ch in word if ch not in punct_prof) for word in tokens]

    # Format Tokens for Vectorization
    preproc_lyrics = " ".join(tokens)

    # Vectorize new lyrics
    lyric_vec = vectorizer.transform([preproc_lyrics])

    # Predict Genres
    pred = model.predict(lyric_vec)

    # Prediction Output
    print(pred)

    prob = model.predict_proba(lyric_vec)
    genres = ["Country", "Pop", "R&B", "Rap", "Rock"]
    result_string = ""
    for i in range(0, 5):
        genre = genres[i]
        genre_i_prob = str(round(prob[i][:,1][0] * 100, 2))
        genre_prob_str = genre_i_prob + "%" + " " + genre + "<br>"
        result_string += genre_prob_str

    # Show results on HTML
    return render_template("prediction.html", prediction_string="Predictions: ", 
                           results= result_string, lyrics = preproc_lyrics)

if __name__ == "__main__":
    app.run()