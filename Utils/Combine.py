import numpy as np
import pandas as pd
import os


def load_data(file_path, columns):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Find where actual data starts
    data_start = next(i for i, line in enumerate(lines) if line.strip() == "@DATA") + 1

    # Read only numerical data
    df = pd.read_csv(file_path, delimiter=',', header=None, names=columns, skiprows=data_start)
    df = df.astype(float)  # Ensure all values are numeric
    return df


def merge_sensor_data(data_dir):
    files = os.listdir(data_dir)
    acc_file = next((f for f in files if '_acc_' in f and '_processed' in f), None)
    gyro_file = next((f for f in files if '_gyro_' in f), None)
    ori_file = next((f for f in files if '_ori_' in f), None)

    if not acc_file or not gyro_file or not ori_file:
        print("Missing required files")
        return

    acc_path = os.path.join(data_dir, acc_file)
    gyro_path = os.path.join(data_dir, gyro_file)
    ori_path = os.path.join(data_dir, ori_file)

    acc_df = load_data(acc_path, ['timestamp', 'acc_x', 'acc_y', 'acc_z'])
    gyro_df = load_data(gyro_path, ['timestamp', 'gyro_x', 'gyro_y', 'gyro_z'])
    ori_df = load_data(ori_path, ['timestamp', 'ori_azimuth', 'ori_pitch', 'ori_roll'])

    merged_df = acc_df.merge(gyro_df, on='timestamp', how='inner')
    merged_df = merged_df.merge(ori_df, on='timestamp', how='inner')

    subject_trial_info = acc_file.replace('acc_', '').replace('_processed.txt', '')
    output_file = os.path.join(data_dir, f"{subject_trial_info}.csv")

    merged_df.to_csv(output_file, sep=',', index=False)
    print(f"Processed {output_file}")



