from flask import Flask, request, jsonify
from sklearn.externals import joblib
import traceback
import pandas as pd
import numpy as np
app = Flask(__name__)
@app.route('/predict', methods=['POST'])
def predict():
    if clf:
        try:
            print('request :',request.json)
            print(request.json)
            json_ = request.json
            prediction = clf.predict(json_)
            return jsonify({'prediction': prediction.tolist()})

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('Model not found')

if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) #command-line argument
    except:
        port = 12345 # If you don't provide any port then the port will be set to 12345
    clf = joblib.load('model.pkl') # Load "model.pkl"
    print ('Model loaded')
    app.run(port=port, debug=True)