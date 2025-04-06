import numpy as np
import pandas as pd
import tensorflow as tf
from Constants import constants
from Utils.Data_loader import load_labels, load_sensor_data
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Paths
sensor_data_path = constants.SENSOR_DATA
label_data_path = constants.LABEL_DATA
sequence_length = 500  # Keep same as MobiFall


def create_sequences(sensor_data, labels, sequence_length):
    sequences = []
    targets = []

    for subject, dataframes in sensor_data.items():
        for df in dataframes:
            filename = df.name  # Get stored filename
            parts = filename.split("T")  # Split on 'T'

            if len(parts) > 1 and "R" in parts[1]:
                task_id = parts[1][:2]  # Extract Task ID
                trial_id = parts[1].split("R")[1][:2]  # Extract Trial ID
            else:
                print(f"Warning: Unexpected filename format {filename}")
                task_id, trial_id = None, None
                continue  # Skip processing if filename is incorrect
            # Find matching label
            label_df = labels.get(subject, pd.DataFrame())
            label_df['Task Code (Task ID)'] = label_df['Task Code (Task ID)'].ffill()
            label_df['Task ID'] = label_df['Task Code (Task ID)'].str.extract(r'\((\d+)\)')
            label_df['Task ID'] = label_df['Task ID'].astype(float)

            label_row = label_df[(label_df['Task ID'] == float(task_id)) & (label_df['Trial ID'] == float(trial_id))]
            fall_frames = label_row[
                ['Fall_onset_frame', 'Fall_impact_frame']].values.flatten() if not label_row.empty else None

            for i in range(0, len(df) - sequence_length, sequence_length // 2):
                seq = df.iloc[i:i + sequence_length, 2:].values
                label = 1 if fall_frames is not None and any(
                    fall_frames[0] <= j <= fall_frames[1] for j in range(i, i + sequence_length)
                ) else 0
                sequences.append(seq)
                targets.append(label)

    return np.array(sequences), np.array(targets)

def process_data_K():

    labels = load_labels(label_data_path)
    sensor_data = load_sensor_data(sensor_data_path)

    # Normalize sensor data
    scaler = StandardScaler()
    for subject in sensor_data:
        for i in range(len(sensor_data[subject])):
            sensor_data[subject][i].iloc[:, 2:] = scaler.fit_transform(sensor_data[subject][i].iloc[:, 2:])
    X, y = create_sequences(sensor_data, labels, sequence_length)

    # Pad sequences
    X = pad_sequences(X, maxlen=sequence_length, dtype='float32', padding='post', truncating='post')

    return X, y
