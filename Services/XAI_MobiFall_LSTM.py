import os
import shap
import pandas as pd
import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from Constants import constants
from Controller.process_all_mobifall import process_all
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


# process_all()

# Hyperparameters
BATCH_SIZE = 16  # Increased batch size for faster training
EPOCHS = 15
MAX_LEN = 500  # Set a reasonable max sequence length


# Function to get max sequence length
def get_max_seq_length(file_paths):
    return min(MAX_LEN, max(len(pd.read_csv(file)) for file in file_paths))


# Custom Data Loader
class FallDataset(tf.keras.utils.Sequence):
    def __init__(self, file_paths, batch_size=BATCH_SIZE, seq_length=None):
        self.file_paths = file_paths
        self.batch_size = batch_size
        self.seq_length = seq_length or get_max_seq_length(file_paths)

    def __len__(self):
        return int(np.ceil(len(self.file_paths) / self.batch_size))

    def __getitem__(self, idx):
        batch_files = self.file_paths[idx * self.batch_size: (idx + 1) * self.batch_size]
        X, y = [], []

        for file in batch_files:
            df = pd.read_csv(file).iloc[:, 1:]  # Drop timestamp
            df = (df - df.mean()) / df.std()  # Normalize
            label = 1 if any(fall_code in file for fall_code in ["FOL", "FKL", "BSC", "SDL"]) else 0

            # Padding
            if len(df) >= self.seq_length:
                df = df.iloc[:self.seq_length, :]
            else:
                padding = np.zeros((self.seq_length - len(df), df.shape[1]))
                df = np.vstack([df, padding])

            X.append(df)
            y.append(label)
        return np.array(X), np.array(y)


# Load dataset
data_path = constants.MOBIFALL_PROCESSED
file_paths = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith(".csv")]
np.random.shuffle(file_paths)
seq_length = get_max_seq_length(file_paths)

# Train-Test Split
split_idx = int(0.8 * len(file_paths))
train_files, val_files = file_paths[:split_idx], file_paths[split_idx:]

# Create Data Loaders
train_dataset = FallDataset(train_files, batch_size=BATCH_SIZE, seq_length=seq_length)
val_dataset = FallDataset(val_files, batch_size=BATCH_SIZE, seq_length=seq_length)

def train_model_mobifall():
    # Model Architecture
    model = Sequential([
        Input(shape=(seq_length, 9)),
        Bidirectional(LSTM(64, return_sequences=True)),
        BatchNormalization(),
        Dropout(0.3),
        Bidirectional(LSTM(32)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    # Compile Model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    # Train Model with Early Stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS, callbacks=[early_stop])

    # Plot Loss
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.show()

    # Save Model
    model.save(r"Services\fall_detection_lstm.h5")
    print("Model saved as 'fall_detection_lstm.h5'")

    X_val, y_val = val_dataset[0]

    # Predict on validation data
    y_pred_mobifall = (model.predict(X_val) > 0.5).astype(int)

    # Generate confusion matrix
    cm_mobifall = confusion_matrix(y_val, y_pred_mobifall)

    # Display confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_mobifall)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix - MobiFall Model")
    plt.show()



def display_result_mobifall():
    # Load the trained LSTM model
    model = load_model(r"Services\fall_detection_lstm.h5")

    # Print expected input shape
    print("Expected model input shape:", model.input_shape)

    # Fetch validation data
    X_val, y_val = val_dataset[0]  # First batch from validation dataset
    X_sample = np.array(X_val[:5])  # Take first 5 samples

    # Debugging: Check input shape
    print(f"X_sample shape: {X_sample.shape}")  # Should be (5, 500, 9)

    # -----------------------------
    # SHAP Explainer
    # -----------------------------
    explainer = shap.GradientExplainer(model, X_sample)

    # Compute SHAP values
    shap_values = explainer.shap_values(X_sample)

    # Convert to NumPy array if it's a list (for multi-output models)
    if isinstance(shap_values, list):
        shap_values = np.array(shap_values[0])

    # Debugging: Check SHAP values shape
    print(f"shap_values shape: {shap_values.shape}")  # Expected: (5, 500, 9)

    # --------------------------------------
    # 2️⃣ SHAP Heatmap (Importance Over Time)
    # --------------------------------------
    # Compute mean SHAP values over all samples
    shap_mean_all = np.mean(np.abs(shap_values), axis=0)  # Shape: (500, 9)

    # Fix: Remove extra dimension if present
    shap_mean_all = np.squeeze(shap_mean_all)  # Ensures shape (500, 9)

    # Double-check final shape
    print(f"shap_mean_all shape: {shap_mean_all.shape}")  # Should be (500, 9)

    # Plot heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(shap_mean_all.T, cmap="coolwarm", xticklabels=50,
                yticklabels=["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z",
                             "ori_azimuth", "ori_pitch", "ori_roll"])
    plt.xlabel("Time Step")
    plt.ylabel("Feature")
    plt.title("Overall SHAP Feature Importance Across Time")
    plt.show()


