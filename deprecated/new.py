import pandas as pd
import numpy as np
import librosa
import os
from sklearn.model_selection import train_test_split # Or use the provided split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences # Using TensorFlow/Keras example

# --- Constants ---
DATA_DIR = '/Users/manjunathanr/Documents/Personal/Heartbeat/ICBHI_final_database'
AUDIO_DIR = os.path.join(DATA_DIR, 'audio_and_txt_files') # Assuming audio and txt are together
ANNOTATIONS_DIR = AUDIO_DIR
TRAIN_TEST_LIST = os.path.join(DATA_DIR, 'train_test.txt')
MAX_SEQ_LENGTH = 500 # Example: Max number of MFCC frames per cycle, adjust based on data
N_MFCC = 20          # Example: Number of MFCC features
HOP_LENGTH = 512     # Example
N_FFT = 2048         # Example
TARGET_SR = 44100    # Example: Resample all audio to this rate

# --- 1. Load Train/Test Split Info ---
split_df = pd.read_csv(TRAIN_TEST_LIST, sep='\t', header=None, names=['filename_base', 'split'])
train_files = set(split_df[split_df['split'] == 'train']['filename_base'])
test_files = set(split_df[split_df['split'] == 'test']['filename_base'])

# --- 2 & 3. Load Annotations, Segment Audio, Create Labels ---
all_features = []
all_labels = []
all_metadata = [] # To store associated metadata for later use
file_assignment = [] # Keep track of train/test for each segment

print("Processing audio files...")
for filename in os.listdir(AUDIO_DIR):
    if filename.endswith(".wav"):
        base_name = filename.replace(".wav", "")
        txt_file = os.path.join(ANNOTATIONS_DIR, base_name + ".txt")
        wav_file = os.path.join(AUDIO_DIR, filename)

        if not os.path.exists(txt_file):
            print(f"Warning: Annotation file not found for {filename}")
            continue

        # Determine if train or test
        if base_name in train_files:
            current_split = 'train'
        elif base_name in test_files:
            current_split = 'test'
        else:
            print(f"Warning: File {base_name} not found in train_test.txt")
            continue # Skip files not in the split list

        # Parse filename for metadata (example)
        parts = base_name.split('_')
        patient_id, rec_index, chest_loc, acq_mode, equipment = parts[0], parts[1], parts[2], parts[3], parts[4]
        metadata = {
            'patient_id': patient_id,
            'rec_index': rec_index,
            'chest_loc': chest_loc,
            'acq_mode': acq_mode,
            'equipment': equipment,
            'filename': base_name
        }

        annotations = pd.read_csv(txt_file, sep='\t', header=None, names=['start', 'end', 'crackles', 'wheezes'])

        try:
            y, sr = librosa.load(wav_file, sr=TARGET_SR) # Load and resample
        except Exception as e:
            print(f"Error loading {wav_file}: {e}")
            continue

        for index, row in annotations.iterrows():
            start_sample = librosa.time_to_samples(row['start'], sr=sr)
            end_sample = librosa.time_to_samples(row['end'], sr=sr)
            segment = y[start_sample:end_sample]

            if len(segment) == 0:
                continue # Skip empty segments

            # --- 4. Extract Features ---
            mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
            mfccs = mfccs.T # Transpose to (time_steps, n_mfcc)

            if mfccs.shape[0] > 0: # Ensure MFCCs were extracted
                 all_features.append(mfccs)
                 # Multi-label: [crackles, wheezes]
                 all_labels.append([row['crackles'], row['wheezes']])
                 all_metadata.append(metadata)
                 file_assignment.append(current_split)

print(f"Processed {len(all_features)} respiratory cycles.")

# --- 5. Handle Variable Lengths ---
# Check sequence lengths
# lengths = [f.shape[0] for f in all_features]
# print(f"Max sequence length: {np.max(lengths)}, 95th percentile: {np.percentile(lengths, 95)}")
# Choose MAX_SEQ_LENGTH based on this

features_padded = pad_sequences(all_features, maxlen=MAX_SEQ_LENGTH, padding='post', truncating='post', dtype='float32')
labels = np.array(all_labels)

# --- 6. Split Data ---
train_indices = [i for i, split in enumerate(file_assignment) if split == 'train']
test_indices = [i for i, split in enumerate(file_assignment) if split == 'test']

X_train = features_padded[train_indices]
y_train = labels[train_indices]
X_test = features_padded[test_indices]
y_test = labels[test_indices]

# Store metadata associated with test set for later analysis if needed
metadata_test = [all_metadata[i] for i in test_indices]

print(f"Train set shape: {X_train.shape}, Labels: {y_train.shape}")
print(f"Test set shape: {X_test.shape}, Labels: {y_test.shape}")

# --- 7. Normalize Features ---
# Reshape for StandardScaler (needs 2D input: samples x features)
# Treat each time step's MFCC vector as features independently? Or normalize across time?
# Simpler: Normalize each MFCC coefficient across all time steps and samples.
scaler = StandardScaler()

# Reshape X_train to (num_samples * MAX_SEQ_LENGTH, N_MFCC)
nsamples, nsteps, nfeatures = X_train.shape
X_train_reshaped = X_train.reshape(-1, nfeatures)

# Fit scaler ONLY on training data
scaler.fit(X_train_reshaped)

# Transform train data
X_train_scaled_reshaped = scaler.transform(X_train_reshaped)
X_train_scaled = X_train_scaled_reshaped.reshape(nsamples, nsteps, nfeatures)

# Transform test data
nsamples_test, nsteps_test, nfeatures_test = X_test.shape
X_test_reshaped = X_test.reshape(-1, nfeatures_test)
X_test_scaled_reshaped = scaler.transform(X_test_reshaped)
X_test_scaled = X_test_scaled_reshaped.reshape(nsamples_test, nsteps_test, nfeatures_test)

print("Data preparation complete.")

# Save preprocessed data and scaler (optional but recommended)
# np.savez('preprocessed_data.npz', X_train=X_train_scaled, y_train=y_train, X_test=X_test_scaled, y_test=y_test, metadata_test=metadata_test)
# import joblib
# joblib.dump(scaler, 'scaler.joblib')