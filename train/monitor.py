import pandas as pd
from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection import GridSearchCV
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import joblib
import subprocess
import os
import warnings
import virtualenv
warnings.filterwarnings("ignore")

def load_model(model_name):
    model=joblib.load(model_name)
    return model

def monitor(data_path):

    df_test = pd.read_csv(data_path+'test.csv')

    X_test = df_test[['hour_of_day','day_of_week','year']]
    y_test = df_test['Reading']

    def evaluate(model, X_test, y_test):
        predictions = model.predict(X_test)
        errors = abs(predictions - y_test)
        mape = 100 * np.mean(errors / y_test)
        accuracy = 100 - mape
        return accuracy
    
    # Load model
    model = load_model("./webApp/model.pkl")

    # Log metrics
    mse = evaluate(model, X_test, y_test)
    if mse > 80.0:
        print("Model accuracy is above threshold")
        print("mse:: ", mse)
    else:
        print("Model accuracy is under  threshold")
        print("mse::", mse)

        # Set execution policy (if necessary)
        # subprocess.call(['powershell', 'Set-ExecutionPolicy', 'RemoteSigned', '-Scope', 'Process'])

        # # Activate virtual environment (if using 'activate' script)
        # subprocess.call(['venv\Scripts\activate'])

        # Execute training script

        subprocess.call(['python', 'train/train.py'])


if __name__=="__main__":
    data_path = "./data/prepared/"

    monitor(data_path)