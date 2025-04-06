import os
import shutil
from Constants import constants
from Utils.Interpolate import process_acceleration
from Utils.Combine import merge_sensor_data

main_dir = r"C:\Users\KIIT\PycharmProjects\Fall Detection\MobiFall_Dataset_v2.0"

def process_all():
    output_dir = os.path.join(main_dir, "Output")
    os.makedirs(output_dir, exist_ok=True)

    for sub in range(1, 15):  # Loop through sub1 to sub14
        sub_path = os.path.join(main_dir, f"sub{sub}")
        if not os.path.exists(sub_path):
            continue  # Skip if subdirectory does not exist

        for activity in os.listdir(sub_path):  # Iterate through FALLS or other activities
            activity_path = os.path.join(sub_path, activity)
            if not os.path.isdir(activity_path):
                continue

            for sub_activity in os.listdir(activity_path):  # Iterate through BSC, FKL, etc.
                sub_activity_path = os.path.join(activity_path, sub_activity)
                if not os.path.isdir(sub_activity_path):
                    continue

                for folder in os.listdir(sub_activity_path):  # Iterate through 1_1, 1_2, etc.
                    folder_path = os.path.join(sub_activity_path, folder)
                    if not os.path.isdir(folder_path):
                        continue

                    # Step 1: Interpolate accelerometer data
                    process_acceleration(folder_path)

                    # Step 2: Merge sensor data
                    merge_sensor_data(folder_path)

                    # Step 3: Move final output files to Output directory
                    for file in os.listdir(folder_path):
                        if file.endswith(".csv"):  # Move only final merged CSV files
                          shutil.move(os.path.join(folder_path, file), os.path.join(output_dir, file))