from flask import Flask, request, jsonify, render_template, send_from_directory
import tensorflow as tf
import librosa
import numpy as np
import joblib
import os
import re

# --- Constants ---
SAVED_ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_assets')
MODEL_PATH = os.path.join(SAVED_ASSETS_DIR, 'best_model.keras')
SCALER_PATH = os.path.join(SAVED_ASSETS_DIR, 'scaler.joblib')
STATS_PATH = os.path.join(SAVED_ASSETS_DIR, 'dataset_stats.joblib')
DIAGNOSIS_MAP_PATH = os.path.join(SAVED_ASSETS_DIR, 'patient_diagnosis_map.joblib')
TARGET_SR = 22050
N_MFCC = 20
MAX_SEQ_LENGTH = 500  # Must match training
N_FFT = 2048
HOP_LENGTH = 512
WINDOW_DURATION_S = 2.0  # Duration of analysis window
HOP_DURATION_S = 1.0  # Step between windows

# --- Initialize App ---
app = Flask(__name__, static_folder='static', template_folder='templates')

# --- Load Model and Scaler ---
try:
    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Loading scaler...")
    scaler = joblib.load(SCALER_PATH)
    print("Loading dataset stats...")
    dataset_stats = joblib.load(STATS_PATH)
    print("Loading patient diagnosis map...")
    patient_diagnosis_map = joblib.load(DIAGNOSIS_MAP_PATH)
    print("Resources loaded successfully.")
except Exception as e:
    print(f"Error loading resources: {str(e)}")
    print("Please run scripts/prepare_data.py and scripts/train_model.py first.")
    model = None
    scaler = None
    dataset_stats = {
        "overall_prevalence": {"crackles": 0, "wheezes": 0},
        "prevalence_by_location": {},
        "prevalence_by_equipment": {},
        "prevalence_by_diagnosis": {}
    }
    patient_diagnosis_map = {}

# --- Helper function for processing audio ---
def process_audio_window(y_window, sr):
    mfccs = librosa.feature.mfcc(y=y_window, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mfccs = mfccs.T  # (time_steps, n_mfcc)

    # Pad/Truncate
    if mfccs.shape[0] < MAX_SEQ_LENGTH:
        pad_width = MAX_SEQ_LENGTH - mfccs.shape[0]
        mfccs = np.pad(mfccs, ((0, pad_width), (0, 0)), mode='constant')
    elif mfccs.shape[0] > MAX_SEQ_LENGTH:
        mfccs = mfccs[:MAX_SEQ_LENGTH, :]

    # Scale
    mfccs_reshaped = mfccs.reshape(-1, N_MFCC)
    mfccs_scaled_reshaped = scaler.transform(mfccs_reshaped)  # Use transform, not fit!
    mfccs_scaled = mfccs_scaled_reshaped.reshape(1, MAX_SEQ_LENGTH, N_MFCC)  # Add batch dimension

    return mfccs_scaled

# --- Function to extract patient ID from filename ---
def extract_patient_id_from_filename(filename):
    # Try to extract patient ID from filename patterns like 101_1b1_Al_sc_Meditron.wav
    match = re.match(r'^(\d+)_', filename)
    if match:
        return match.group(1)
    return None

# --- Main route for the UI ---
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# --- Static files ---
@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

# --- Prediction Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({"error": "Model or scaler not loaded. Run prepare_data.py and train_model.py first."}), 500
        
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    file = request.files['audio']
    patient_id = request.form.get('patient_id', None)
    
    # If no patient ID was provided, try to extract it from filename
    if not patient_id:
        patient_id = extract_patient_id_from_filename(file.filename)
    
    # Get diagnosis if patient ID is available
    diagnosis = None
    if patient_id:
        diagnosis = patient_diagnosis_map.get(patient_id, "Unknown")

    try:
        # Load audio using librosa directly from file object
        y, sr = librosa.load(file, sr=TARGET_SR)

        window_samples = int(WINDOW_DURATION_S * sr)
        hop_samples = int(HOP_DURATION_S * sr)

        predictions = []
        num_windows = max(1, (len(y) - window_samples) // hop_samples + 1)

        # Initial status update
        status_update = {
            "status": "processing",
            "message": "Starting audio analysis...",
            "progress": 0,
            "total_windows": num_windows
        }
        
        for i in range(num_windows):
            start_sample = i * hop_samples
            end_sample = min(start_sample + window_samples, len(y))
            
            # Skip if window is too small
            if end_sample - start_sample < sr * 0.5:  # At least 0.5 second
                continue
                
            y_window = y[start_sample:end_sample]

            start_time = start_sample / sr
            end_time = end_sample / sr

            # Process window and predict
            features = process_audio_window(y_window, sr)
            proba = model.predict(features, verbose=0)[0]  # Get prediction for the single window

            predictions.append({
                "start": round(start_time, 2),
                "end": round(end_time, 2),
                "crackles_prob": round(float(proba[0]), 3),
                "wheezes_prob": round(float(proba[1]), 3)
            })

        # Calculate overall prediction statistics
        avg_crackles = np.mean([p["crackles_prob"] for p in predictions])
        avg_wheezes = np.mean([p["wheezes_prob"] for p in predictions])
        max_crackles = np.max([p["crackles_prob"] for p in predictions])
        max_wheezes = np.max([p["wheezes_prob"] for p in predictions])
        
        result = {
            "predictions": predictions,
            "summary": {
                "avg_crackles_prob": round(float(avg_crackles), 3),
                "avg_wheezes_prob": round(float(avg_wheezes), 3),
                "max_crackles_prob": round(float(max_crackles), 3),
                "max_wheezes_prob": round(float(max_wheezes), 3),
                "patient_id": patient_id,
                "diagnosis": diagnosis
            }
        }
        
        # If there's a diagnosis, include relevant statistics
        if diagnosis and diagnosis in dataset_stats["prevalence_by_diagnosis"]:
            diag_stats = dataset_stats["prevalence_by_diagnosis"][diagnosis]
            result["diagnosis_stats"] = {
                "name": diagnosis,
                "crackles_prevalence": diag_stats["crackles"],
                "wheezes_prevalence": diag_stats["wheezes"],
                "sample_count": diag_stats.get("count", 0)
            }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"Error processing audio: {str(e)}"}), 500

# --- Stats Endpoint ---
@app.route('/stats', methods=['GET'])
def get_stats():
    return jsonify(dataset_stats)

# --- Diagnosis Info Endpoint ---
@app.route('/diagnoses', methods=['GET'])
def get_diagnoses():
    if not dataset_stats or "prevalence_by_diagnosis" not in dataset_stats:
        return jsonify({"error": "Diagnosis data not available"}), 404
    
    return jsonify(dataset_stats["prevalence_by_diagnosis"])

if __name__ == '__main__':
    # Use '0.0.0.0' to make it accessible on your network if needed
    # Debug should be False in production
    app.run(host='127.0.0.1', port=5000, debug=True) 