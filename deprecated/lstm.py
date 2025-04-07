import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report, roc_auc_score

# --- Assuming X_train_scaled, y_train, X_test_scaled, y_test are loaded ---
INPUT_SHAPE = (MAX_SEQ_LENGTH, N_MFCC)
NUM_CLASSES = 2 # Crackles, Wheezes

# --- 1. Define Model Architecture ---
model = Sequential([
    Input(shape=INPUT_SHAPE),
    Bidirectional(LSTM(64, return_sequences=True)), # Example: 64 units
    Dropout(0.3),
    Bidirectional(LSTM(64, return_sequences=False)), # Last LSTM layer
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(NUM_CLASSES, activation='sigmoid') # Sigmoid for multi-label binary output
])

# --- 2. Compile Model ---
model.compile(optimizer='adam',
              loss='binary_crossentropy', # Correct loss for multi-label
              metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]) # Add AUC

model.summary()

# --- 3. Train Model ---
BATCH_SIZE = 32
EPOCHS = 50 # Adjust as needed

# Callbacks
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_auc', mode='max', save_best_only=True, verbose=1)
early_stopping = EarlyStopping(monitor='val_auc', mode='max', patience=10, verbose=1) # Stop if val_auc doesn't improve for 10 epochs

history = model.fit(X_train_scaled, y_train,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    validation_data=(X_test_scaled, y_test),
                    callbacks=[checkpoint, early_stopping])

# --- 4. Evaluate Model ---
# Load the best model saved by ModelCheckpoint
best_model = tf.keras.models.load_model('best_model.keras')

loss, accuracy, auc = best_model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"\nTest Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test AUC: {auc:.4f}")

# Detailed evaluation
y_pred_proba = best_model.predict(X_test_scaled)
y_pred = (y_pred_proba > 0.5).astype(int) # Threshold probabilities

print("\nClassification Report:")
# Note: classification_report works best label-wise for multi-label
report = classification_report(y_test, y_pred, target_names=['Crackles', 'Wheezes'], zero_division=0)
print(report)

# Calculate AUC per class if needed
auc_crackles = roc_auc_score(y_test[:, 0], y_pred_proba[:, 0])
auc_wheezes = roc_auc_score(y_test[:, 1], y_pred_proba[:, 1])
print(f"Test AUC Crackles: {auc_crackles:.4f}")
print(f"Test AUC Wheezes: {auc_wheezes:.4f}")


# --- 5. Save Final Model (optional, if different from best checkpoint) ---
# best_model.save('final_respiratory_sound_model.keras')
# Save the scaler too!
# import joblib
# joblib.dump(scaler, 'scaler.joblib')