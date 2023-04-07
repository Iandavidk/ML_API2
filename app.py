import numpy as np
from sklearn.datasets import load_iris
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__) 
model = pickle.load(open('kmeans_model.pkl', 'rb'))
iris = load_iris()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''
    Render the prediction results on HTML
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = iris.target_names[prediction[0]]

    return render_template('index.html', prediction_text='Iris type should be {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)