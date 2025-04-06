import numpy as np
import tensorflow as tf
import shap
import seaborn as sns
import matplotlib.pyplot as plt
from Controller.process_data_kfall import process_data_K
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


# Hyperparameters
BATCH_SIZE = 16
EPOCHS = 15
SEQ_LENGTH = 500

# Split dataset
X, y = process_data_K()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def build_model():
    model = Sequential([
        Input(shape=(SEQ_LENGTH, 9)),
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
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model
# Train Model
def train_model_k():
    model = build_model()
    model.summary()
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_test, y_test), callbacks=[early_stop])
    model.save("kfall_lstm.h5")
    print("Model saved as 'kfall_lstm.h5'")

    # Plot Loss
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.show()

    y_pred_kfall = (model.predict(X_test) > 0.5).astype(int)

    # Generate confusion matrix
    cm_kfall = confusion_matrix(y_test, y_pred_kfall)

    # Display confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_kfall)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix - KFall Model")
    plt.show()


# Explainability (SHAP)
def explain_model_k():
    model = load_model("kfall_lstm.h5")
    X_sample = X_test[:5]
    print("X_sample shape:", X_sample.shape)
    explainer = shap.GradientExplainer(model, X_sample)
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        shap_values = np.array(shap_values[0])

    shap_mean_all = np.mean(np.abs(shap_values), axis=0)
    shap_mean_all = np.squeeze(shap_mean_all)
    print("SHAP values shape:", shap_mean_all.shape)

    plt.figure(figsize=(12, 6))
    sns.heatmap(shap_mean_all.T, cmap="coolwarm", xticklabels=50,
                yticklabels=["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z", "ori_azimuth", "ori_pitch",
                             "ori_roll"])
    plt.xlabel("Time Step")
    plt.ylabel("Feature")
    plt.title("Overall SHAP Feature Importance Across Time")
    plt.show()