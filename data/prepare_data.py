# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# Function to normalize data
def normalize_data(dataframe):
    # Initialize a MinMaxScaler
    scaler = preprocessing.MinMaxScaler()
    # Scale the 'Reading' column
    scaled_reading = scaler.fit_transform(dataframe['Reading'].values.reshape(-1, 1))
    dataframe['Reading'] = scaled_reading
    # Drop missing values and unnecessary columns
    dataframe = dataframe.dropna()
    dataframe = dataframe.drop(columns=['Machine_ID','Sensor_ID'])
    # Convert 'Timestamp' to datetime format and create new columns
    dataframe['Timestamp'] = pd.to_datetime(dataframe['Timestamp'])
    dataframe['hour_of_day'] = dataframe['Timestamp'].dt.hour
    dataframe['day_of_week'] = dataframe['Timestamp'].dt.dayofweek
    dataframe['year'] = dataframe['Timestamp'].dt.year
    return dataframe

# Function to split data into training and testing sets
def split_data(dataframe, test_size):
    # Separate features and label
    features = dataframe.drop('Reading', axis=1)
    label = dataframe['Reading']
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=test_size, random_state=41)
    # Combine features and labels into DataFrames
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    return train_df, test_df


# Entry point of the script
if __name__ == "__main__":
    # Define file paths
    train_output_path = "./data/prepared/train.csv"
    test_output_path = "./data/prepared/test.csv"
    input_file_path = "./data/dummy_sensor_data.csv"
    # Load data
    dataframe = pd.read_csv(input_file_path)
    # Normalize data
    dataframe = normalize_data(dataframe)
    # Split data
    train_df, test_df = split_data(dataframe, test_size=0.2)
    # Write DataFrames to CSV files
    train_df.to_csv(train_output_path)
    test_df.to_csv(test_output_path)