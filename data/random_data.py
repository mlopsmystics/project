from datetime import datetime, timedelta
import subprocess
import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)


# Function to generate dummy data and append to existing CSV
def generate_and_append_data(file_path, num_machines=5, num_sensors=3, freq='H'):
    try:
        existing_data = pd.read_csv(file_path)
    except FileNotFoundError:
        existing_data = pd.DataFrame(columns=['Timestamp', 'Machine_ID', 'Sensor_ID', 'Reading'])

    if not existing_data.empty and 'Timestamp' in existing_data.columns:
        existing_data['Timestamp'] = pd.to_datetime(existing_data['Timestamp'])
        start_date = existing_data['Timestamp'].max() + timedelta(hours=1)
    else:
        start_date = datetime.now()

    end_date = start_date + timedelta(days=1)

    new_data = generate_dummy_data(start_date, end_date, num_machines, num_sensors, freq)
    updated_data = pd.concat([existing_data, new_data], ignore_index=True)

    updated_data.to_csv(file_path, index=False)



# Function to generate random datetime within a given range
def random_dates(start_date, end_date, n=10):
    date_range = (end_date - start_date).days
    random_dates = [start_date + timedelta(days=np.random.randint(date_range)) for _ in range(n)]
    return sorted(random_dates)

# Function to generate dummy data
def generate_dummy_data(start_date, end_date, num_machines=5, num_sensors=3, freq='H'):
    machine_ids = [f'Machine_{i}' for i in range(1, num_machines + 1)]
    sensor_ids = [f'Sensor_{j}' for j in range(1, num_sensors + 1)]

    dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    data = {'Timestamp': [], 'Machine_ID': [], 'Sensor_ID': [], 'Reading': []}

    for date in dates:
        for machine_id in machine_ids:
            for sensor_id in sensor_ids:
                data['Timestamp'].append(date)
                data['Machine_ID'].append(machine_id)
                data['Sensor_ID'].append(sensor_id)
                # Simulate sensor readings as random values
                data['Reading'].append(np.random.normal(loc=100, scale=20))

    return pd.DataFrame(data)


if __name__ == "__main__":
    data_file_path = 'dummy_sensor_data.csv'
    # Define date range for dummy data
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 1, 10)

    # # Generate dummy data
    # dummy_data = generate_dummy_data(start_date, end_date, num_machines=5, num_sensors=3)

    # # Save dummy data to CSV file
    # dummy_data.to_csv('dummy_sensor_data.csv', index=False)

    generate_and_append_data(data_file_path)
