from ast import literal_eval
import numpy as np
import pandas as pd
import pickle
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


#import data
# Read in preprocessed data
preproc_df = pd.read_csv("data/genre_prepped.csv.gz", compression = "gzip",
                         converters = {"tokens": literal_eval, "genre" : literal_eval})

# drop unnecessary columns (index and unnamed index columns)
preproc_df = preproc_df.drop(preproc_df.columns[0:2], axis = 1)
print("Loaded Data")

# Convert genres to set for multilabel encoding
preproc_df["genre_set"] = preproc_df["genre"].map(set)

mlb = MultiLabelBinarizer()
genre_array = mlb.fit_transform(preproc_df["genre_set"])
# Convert label array to sparse matrix
genre_sparse = sparse.csr_matrix(genre_array)
print("Binarized Output")

print("Beginning TF-IDF Vectorization")
tfidf_vectorizer = TfidfVectorizer().fit(preproc_df['lyrics_clean'])
text_sparse = tfidf_vectorizer.transform(preproc_df['lyrics_clean'])
print("TF-IDF Vectorization Complete")

print("Partitioning Data for Model Training")
X_train, X_test, y_train, y_test = train_test_split(
    text_sparse, genre_array,
    test_size= 0.3 , random_state= 2023)

print("Training Model")
svc_clf = MultiOutputClassifier(
    SVC(
        C = 100, kernel = "linear", 
        probability = True, random_state = 2023)
    ).fit(X_train, y_train)

print("Predicting Test Set")
y_pred = svc_clf.predict(X_test)

# Save model
print("Pickling Model")
pickle.dump(svc_clf, open('templates/model.pkl', 'wb'))
print("Picking Vectorizer")
pickle.dump(tfidf_vectorizer, open("templates/vectorizer.pkl", "wb"))
