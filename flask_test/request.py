import requests
import pickle
url = 'http://localhost:5000/api'

# Error vectorizer does not carry over transform function
vectorizer = pickle.load(open('flask_test/vectorizer.pkl','rb'))

new_lyrics = "test lyrics here"

lyric_vec = vectorizer.transform([new_lyrics])

r = requests.post(url, json={'lyrics': lyric_vec,})
print(r.json())