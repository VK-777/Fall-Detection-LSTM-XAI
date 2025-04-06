import numpy as np
import pandas as pd
import os
from scipy.interpolate import interp1d


def load_data(file_path, columns):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Find where actual data starts
    data_start = next(i for i, line in enumerate(lines) if line.strip() == "@DATA") + 1

    # Read only numerical data
    df = pd.read_csv(file_path, delimiter=',', header=None, names=columns, skiprows=data_start)
    df = df.astype(float)  # Ensure all values are numeric
    return df


def interpolate_acc(acc_df, target_timestamps):
    interpolated = {}
    for axis in ['x', 'y', 'z']:
        f = interp1d(acc_df['timestamp'], acc_df[axis], kind='linear', fill_value='extrapolate')
        interpolated[axis] = f(target_timestamps)
    return pd.DataFrame(
        {'timestamp': target_timestamps, 'x': interpolated['x'], 'y': interpolated['y'], 'z': interpolated['z']})


def process_acceleration(data_dir):
    files = os.listdir(data_dir)
    acc_file = next((f for f in files if '_acc' in f), None)
    gyro_file = next((f for f in files if '_gyro' in f), None)

    if not acc_file or not gyro_file:
        print("Missing required files")
        return

    acc_path = os.path.join(data_dir, acc_file)
    gyro_path = os.path.join(data_dir, gyro_file)

    acc_df = load_data(acc_path, ['timestamp', 'x', 'y', 'z'])
    gyro_df = load_data(gyro_path, ['timestamp', 'x', 'y', 'z'])

    interpolated_acc = interpolate_acc(acc_df, gyro_df['timestamp'])
    output_file = os.path.join(data_dir, acc_file.replace('.txt', '_processed.txt'))

    with open(output_file, 'w') as file:
        file.write("@DATA\n")  # Add @DATA at the top

    interpolated_acc.to_csv(output_file, sep=',', index=False, header=False, mode='a')
    print(f"Processed {output_file}")