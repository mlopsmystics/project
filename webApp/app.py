from flask import Flask, request, jsonify
from datetime import datetime
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
from mlflow.tracking import MlflowClient



app = Flask(__name__)

def load_model(model_name):
    client = MlflowClient()
    model_metadata = client.get_latest_versions(model_name, stages=["None"])
    latest_model_version = model_metadata[0].version
    model = mlflow.pyfunc.load_model(f"models:/{model_name}/{latest_model_version}")
    return model

# Load your trained model
model = load_model('rf-model')

@app.route('/predict', methods=['POST'])
def predict():
    # Get date from the POST request
    data = request.get_json(force=True)
    date_str = data['date']

    # Convert string to datetime object
    date = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')

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
    app.run(debug=True)