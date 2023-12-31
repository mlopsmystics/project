from flask import Flask, request, jsonify, render_template
from datetime import datetime
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
from mlflow.tracking import MlflowClient
import mlflow as mlflow



app = Flask(__name__)

model = joblib.load('./webApp/model.pkl')

@app.route('/', methods=['GET'])
def index():
    # Render the HTML page when the user accesses the root URL
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get date from the POST request
    data = request.get_json(force=True)
    date_str = data['date']

    # Convert string to datetime object
    date = datetime.strptime(date_str, '%Y-%m-%dT%H:%M')

    # Extract features
    hour_of_day = date.hour
    day_of_week = date.weekday()
    year = date.year

    # Create a DataFrame from the features
    df = pd.DataFrame({
        'hour_of_day': [hour_of_day],
        'day_of_week': [day_of_week],
        'year': [year]
    })

    # Make prediction
    prediction = model.predict(df)

    # Return prediction
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)