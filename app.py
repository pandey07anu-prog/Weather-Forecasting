from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    humidity = float(request.form['humidity'])
    windspeed = float(request.form['windspeed'])
    pressure = float(request.form['pressure'])

    data = np.array([[humidity, windspeed, pressure]])

    prediction = model.predict(data)

    result = round(prediction[0], 2)

    return render_template(
        'index.html',
        prediction_text=f'Predicted Temperature: {result} °C'
    )

if __name__ == "__main__":
    app.run(debug=True)