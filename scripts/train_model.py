import tensorflow as tf
import numpy as np
import os
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report, roc_auc_score

# --- Constants ---
SAVED_ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'saved_assets')
MODEL_PATH = os.path.join(SAVED_ASSETS_DIR, 'best_model.keras')
DATA_PATH = os.path.join(SAVED_ASSETS_DIR, 'preprocessed_data.npz')
MAX_SEQ_LENGTH = 500  # Must match prepare_data.py
N_MFCC = 20           # Must match prepare_data.py

# Check if data exists
if not os.path.exists(DATA_PATH):
    print(f"Error: Preprocessed data not found at {DATA_PATH}")
    print("Please run prepare_data.py first.")
    exit(1)

# --- Load preprocessed data ---
print(f"Loading preprocessed data from {DATA_PATH}...")
data = np.load(DATA_PATH)
X_train_scaled = data['X_train']
y_train = data['y_train']
X_test_scaled = data['X_test']
y_test = data['y_test']

print(f"Train set shape: {X_train_scaled.shape}, Labels: {y_train.shape}")
print(f"Test set shape: {X_test_scaled.shape}, Labels: {y_test.shape}")

# --- Define Model Architecture ---
INPUT_SHAPE = (MAX_SEQ_LENGTH, N_MFCC)
NUM_CLASSES = 2  # Crackles, Wheezes

print("Building model...")
model = Sequential([
    Input(shape=INPUT_SHAPE),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.3),
    Bidirectional(LSTM(64, return_sequences=False)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(NUM_CLASSES, activation='sigmoid')  # Sigmoid for multi-label binary output
])

# --- Compile Model ---
model.compile(optimizer='adam',
              loss='binary_crossentropy',  # Correct loss for multi-label
              metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

model.summary()

# --- Train Model ---
BATCH_SIZE = 32
EPOCHS = 50  # Adjust as needed

# Callbacks
checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_auc', mode='max', save_best_only=True, verbose=1)
early_stopping = EarlyStopping(monitor='val_auc', mode='max', patience=10, verbose=1)

print("Starting training...")
history = model.fit(X_train_scaled, y_train,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    validation_data=(X_test_scaled, y_test),
                    callbacks=[checkpoint, early_stopping])

# --- Evaluate Model ---
print("Loading best model for evaluation...")
best_model = tf.keras.models.load_model(MODEL_PATH)

loss, accuracy, auc = best_model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"\nTest Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test AUC: {auc:.4f}")

# Detailed evaluation
y_pred_proba = best_model.predict(X_test_scaled)
y_pred = (y_pred_proba > 0.5).astype(int)  # Threshold probabilities

print("\nClassification Report:")
report = classification_report(y_test, y_pred, target_names=['Crackles', 'Wheezes'], zero_division=0)
print(report)

# Calculate AUC per class
auc_crackles = roc_auc_score(y_test[:, 0], y_pred_proba[:, 0])
auc_wheezes = roc_auc_score(y_test[:, 1], y_pred_proba[:, 1])
print(f"Test AUC Crackles: {auc_crackles:.4f}")
print(f"Test AUC Wheezes: {auc_wheezes:.4f}")

print(f"\nTraining complete. Model saved to {MODEL_PATH}") 