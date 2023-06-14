import numpy as np
from flask import Flask, request, jsonify
import pickle

# Flask instance
app = Flask(__name__)
model = pickle.load(open('flask_test/model.pkl','rb'))

@app.route('/api',methods=['POST'])
def predict():
    # Get the data from the POST request.
    data = request.get_json(force=True)
    # Make prediction using model loaded from disk as per the data.
    prediction = model.predict([[np.array(data['lyrics'])]])
    # Take the first value of prediction
    output = prediction[0]
    return jsonify(output)
if __name__ == '__main__':
    app.run(port=5000, debug=True)

print("Server file initialized")