import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split

def check_data_quality(data_df, dataset_name):
    # Check data quality
    data_quality = {
        "dataset": dataset_name,
        "total_rows": len(data_df),
        "feature_columns": list(data_df.columns[:-1]),
        "target_column": "Reading",
        "null_values": data_df.isnull().sum().to_dict(),
        "mean_Reading": data_df["Reading"].mean(),
        "std_dev_Reading": data_df["Reading"].std(),
        "min_Reading": data_df["Reading"].min(),
        "max_Reading": data_df["Reading"].max(),
        "total_columns": len(data_df.columns),
        "train_test_split": 0.2,
        "random_state": 41
    }
    return data_quality

def append_to_data_quality_file(data_quality, file_path, current_file_path):
    # Append to data_quality.json
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            existing_data_quality = json.load(f)
        existing_data_quality.append(data_quality)
        with open(file_path, "w") as f:
            json.dump(existing_data_quality, f)
    else:
        with open(file_path, "w") as f:
            json.dump([data_quality], f)
    with open(current_file_path, "w") as f:
        json.dump([data_quality], f)

def main():
    # Load train.csv and test.csv using DVC
    train_file_path = 'data/prepared/train.csv'
    test_file_path = 'data/prepared/test.csv'

    train_df = pd.read_csv(train_file_path)
    test_df = pd.read_csv(test_file_path)

    # Check data quality for train.csv
    train_data_quality = check_data_quality(train_df, "train")
    print("Train Data Quality:", train_data_quality)
    append_to_data_quality_file(train_data_quality, "data/data_quality.json",'data/current_data_quality.json')

    # Check data quality for test.csv
    test_data_quality = check_data_quality(test_df, "test")
    print("Test Data Quality:", test_data_quality)
    append_to_data_quality_file(test_data_quality,"data/data_quality.json", "data/current_data_quality.json")

if __name__ == "__main__":
    main()
