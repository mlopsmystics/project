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

    def evaluate_mae(model, X_test, y_test):
        predictions = model.predict(X_test)
        mae = np.mean(abs(predictions - y_test))  # Calculate mean absolute error
        print("Mean Absolute Error:", mae)
        return mae
    
    # # Load model
    model = load_model("./webApp/model.pkl")

    # # Evaluate model
    mae = evaluate_mae(model, X_test, y_test)
    print("mae:: ", mae)
    if mae > 0.09:
        print("Model accuracy is under threshold so train again")
        print("mae::", mae)
        subprocess.call(['python', 'train/train.py'])
    else:
        print("Model accuracy is good enough")
        print("mae:: ", mae)



if __name__=="__main__":
    data_path = "./data/prepared/"

    monitor(data_path)