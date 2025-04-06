import os
import pandas as pd
from Constants import constants


# Read label files
def load_labels(label_data_path):
    labels = {}
    for file in os.listdir(label_data_path):
        if file.endswith(".xlsx"):
            subject_id = file.split("_")[0]
            df = pd.read_excel(os.path.join(label_data_path, file))
            labels[subject_id] = df
    return labels


# Read sensor data
def load_sensor_data(sensor_data_path):
    sensor_data = {}

    base_path = sensor_data_path  # Update this with your actual path

    for subject in os.listdir(base_path):
        subject_path = os.path.join(base_path, subject)

        if os.path.isdir(subject_path):  # Ensure it's a directory
            sensor_data[subject] = []

            for file in os.listdir(subject_path):
                if file.endswith(".csv"):
                    file_path = os.path.join(subject_path, file)
                    df = pd.read_csv(file_path)
                    df.name = file  # Store filename in DataFrame metadata
                    sensor_data[subject].append(df)

    return sensor_data