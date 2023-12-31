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

    def evaluate(model, X_test, y_test):
        print(X_test, y_test)
        predictions = model.predict(X_test)
        errors = abs(predictions - y_test)
        print("Errors:", errors)
        mape = 100 * np.mean(errors / y_test)
        print("Accuracy:", mape)
        accuracy = 100 - mape
        return accuracy
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
    mse = evaluate(model, X_test, y_test)
    print("Finished Evaluating")
    # Show Results
    print("+++++++++++++++++++++++++++++++++++++++")
    print("Results:")
    print("mse:", mse)
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
    metrics[run_id] = {"mse": mse}


    # Save model
    model_dir = f"./train/model_regsistery/models/run_{run_id}"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    with open(os.path.join(model_dir, "model.pkl"), "wb") as f:
        pickle.dump(model.best_estimator_, f)

    # Save the model to the ./webApp directory
    joblib.dump(model.best_estimator_, './webApp/model.pkl')

    # Register model
    model_info = {
        "model_uri": f"models/run_{run_id}/model.pkl",
        "run_id": run_id,
        "model_params": model.best_params_,
        "dataset": "my dataset",
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
    
if __name__=="__main__":
    data_path = "./data/prepared/"
    train(data_path)