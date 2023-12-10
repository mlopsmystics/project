import pandas as pd
from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection import GridSearchCV
import numpy as np
import mlflow
import mlflow.sklearn
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def train(data_path):

    df = pd.read_csv(data_path+'train.csv')
    df_test = pd.read_csv(data_path+'test.csv')
    X = df[['hour_of_day','day_of_week','year']] 
    y = df['Reading']

    def evaluate(model, X_test, y_test):
        predictions = model.predict(X_test)
        errors = abs(predictions - y_test)
        mape = 100 * np.mean(errors / y_test)
        accuracy = 100 - mape
        return accuracy

    X_test = df_test[['hour_of_day','day_of_week','year']]
    y_test = df_test['Reading']

    # Define hyperparameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 8, 15]
    }

    # Initialize MLflow autologging
    mlflow.sklearn.autolog()

    # Train model with hyperparameter tuning
    rf = RandomForestRegressor()
    model = GridSearchCV(rf, param_grid)

    with mlflow.start_run() as run:
        print("MLflow:")
        model.fit(X, y)
        
        # Log metrics
        mse = evaluate(model, X_test, y_test)
        mlflow.log_metric("mse", mse)
            
        # Register the best model
        mlflow.sklearn.log_model(model.best_estimator_, "rf-model")
        mlflow.sklearn.log_model(model, "rf-model")
        model_uri = mlflow.get_artifact_uri('rf-model')
        mlflow.register_model(model_uri, "ReadingPredictor")
    
    print("Best parameters:", model.best_params_)
    
if __name__=="__main__":
    data_path = "./data/prepared/"
    train(data_path)