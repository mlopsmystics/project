import pandas as pd
from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection import GridSearchCV
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import joblib
import os
import json
import pickle
import datetime
import warnings
warnings.filterwarnings("ignore")


def load_model(model_name):
    client = MlflowClient()
    model_versions = client.search_model_versions(f"name='{model_name}'")
    latest_model_version = model_versions[0].version
    model = mlflow.pyfunc.load_model(f"models:/{model_name}/{latest_model_version}")

    # Save the model to the ./webApp directory
    joblib.dump(model, './webApp/model.pkl')


def train(data_path):

    df = pd.read_csv(data_path+'train.csv')
    df_test = pd.read_csv(data_path+'test.csv')
    X = df[['hour_of_day','day_of_week','year']] 
    y = df['Reading']


    X_test = df_test[['hour_of_day','day_of_week','year']]
    y_test = df_test['Reading']

    def evaluate_mae(model, X_test, y_test):
        predictions = model.predict(X_test)
        mae = np.mean(abs(predictions - y_test))  # Calculate mean absolute error
        print("Mean Absolute Error:", mae)
        return mae
    # Define hyperparameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 8, 20]
    }


    # Train model with hyperparameter tuning
    rf = RandomForestRegressor()
    model = GridSearchCV(rf, param_grid)

    # Train model
    print("Training model...")
    model.fit(X, y)
    print("Finished Training")
    print("++++++++++++++++++++++++++++++++++++++++")
    print("Evaluating model...")
    mae = evaluate_mae(model, X_test, y_test)
    print("Finished Evaluating")
    # Show Results
    print("+++++++++++++++++++++++++++++++++++++++")
    print("Results:")
    print("mae:", mae)
    print("Best parameters:", model.best_params_)
    print("Best score:", model.best_score_)
    print("Best estimator:", model.best_estimator_)
    print("Best index:", model.best_index_)
    print("Scorer function:", model.scorer_)


    # Log metrics
    metrics_file = "./train/model_regsistery/metrics.json"

    run_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S") # returns unique run string for mlflow tracking
    if not os.path.exists(metrics_file):
        with open(metrics_file, "w") as f:
            json.dump({}, f)  # create empty dict

    with open(metrics_file) as f:
        metrics = json.load(f)
    metrics[run_id] = {"mae": mae}


    # Save model
    model_dir = f"./train/model_regsistery/models/run_{run_id}"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    with open(os.path.join(model_dir, "model.pkl"), "wb") as f:
        pickle.dump(model.best_estimator_, f)

    # Read current data quality from csv
    data_quality_file = "./data/current_data_quality.json"
    data_quality = []
    if os.path.exists(data_quality_file):
        with open(data_quality_file) as f:
            data_quality = json.load(f)


    # Register model
    model_info = {
        "model_uri": f"models/run_{run_id}/model.pkl",
        "run_id": run_id,
        "model_params": model.best_params_,
        "dataset": data_quality,
        "metrics": metrics[run_id]
    }

    model_registry_file = "./train/model_regsistery/registered_models.json"
    if not os.path.exists(model_registry_file):
        with open(model_registry_file, "w") as f:
            json.dump([], f)

    with open(model_registry_file) as f:
        model_registry = json.load(f)

    model_registry.append(model_info)
    with open(model_registry_file, "w") as f:
        json.dump(model_registry, f)

    # Load and check the best result from the file
    best_result_file = "./train/best_model.json"
    if os.path.exists(best_result_file):
        with open(best_result_file) as f:
            best_result = json.load(f)
        best_mae = best_result.get("mae", float('inf'))
        
        # Compare with the current result
        if mae < best_mae:
            print("New best result! Updating the best.json file.")
            best_result["mae"] = mae
            with open(best_result_file, "w") as f:
                json.dump(model_info, f)
            # Save the model to the ./webApp directory
            joblib.dump(model.best_estimator_, './webApp/model.pkl')
        else:
            print("Current result is not better than the best result. No update to the best.json file.")
    else:
        # Create the file and store the current result
        print("Creating best.json file with the current result.")
        with open(best_result_file, "w") as f:
            json.dump(model_info, f)
        # Save the model to the ./webApp directory
        joblib.dump(model.best_estimator_, './webApp/model.pkl')

    print("Finished")

    # with mlflow.start_run() as run:
    #     print("MLflow:")
    #     model.fit(X, y)
        
    #     # Log metrics
    #     mse = evaluate(model, X_test, y_test)
    #     mlflow.log_metric("mse", mse)
    #     # set the tracking uri
    #     mlflow.set_tracking_uri("mlruns") 
            
    #     # Register the best model
    #     mlflow.sklearn.log_model(model.best_estimator_, "rf-model")
    #     mlflow.sklearn.log_model(model, "rf-model")
    #     model_uri = mlflow.get_artifact_uri('rf-model')
    #     mlflow.register_model(model_uri, "rf-model")
    
    # print("Best parameters:", model.best_params_)


def trainMlFlow(data_path):

    df = pd.read_csv(data_path+'train.csv')
    df_test = pd.read_csv(data_path+'test.csv')
    X = df[['hour_of_day','day_of_week','year']] 
    y = df['Reading']


    X_test = df_test[['hour_of_day','day_of_week','year']]
    y_test = df_test['Reading']

    def evaluate_mae(model, X_test, y_test):
        predictions = model.predict(X_test)
        mae = np.mean(abs(predictions - y_test))  # Calculate mean absolute error
        print("Mean Absolute Error:", mae)
        return mae
    # Define hyperparameter grid
    param_grid = {
        'n_estimators': [50],
        'max_depth': [5]
    }


    # Train model with hyperparameter tuning
    rf = RandomForestRegressor()
    model = GridSearchCV(rf, param_grid)
    # set the tracking uri
    mlflow.set_tracking_uri("mlruns") 
    mlflow.set_experiment("/rf-model")

    with mlflow.start_run(run_name='hyperparametersTune') as run:
        print("MLflow:")
        model.fit(X, y)
        
        # Log metrics
        mse = evaluate_mae(model, X_test, y_test)
        mlflow.log_metric("mse", mse)
            
        # Register the best model
        mlflow.sklearn.log_model(model.best_estimator_, "rf-model")
        mlflow.sklearn.log_model(model, "rf-model")
        model_uri = mlflow.get_artifact_uri('rf-model')
        mlflow.register_model(model_uri, "rf-model")
    
    print("Best parameters:", model.best_params_)

if __name__=="__main__":
    data_path = "./data/prepared/"
    train(data_path)
    trainMlFlow(data_path)