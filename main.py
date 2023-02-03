import joblib
from flask import Flask, request, redirect, render_template
import numpy as np

app = Flask(__name__)
model = joblib.load('Models/forest_clf.pkl')
scaler = joblib.load('Transformers/scaler.joblib')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = [field for field in request.form.values()]
    data = np.array(data).reshape(1, -1)
    data = scaler.transform(data)
    pred = model.predict(data)
    if pred == 'BIT':
        return render_template('bit.html')
    elif pred == 'SUB':
        return render_template('sub.html')
    elif pred == 'LIG':
        return render_template('lig.html')


@app.route('/predict_again')
def predict_again():
    return redirect('/')


if __name__ == '__main__':
    app.run(debug=True)