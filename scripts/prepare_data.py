import pandas as pd
import numpy as np
import librosa
import os
from sklearn.preprocessing import StandardScaler
import joblib
from tqdm import tqdm

# --- Constants ---
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'raw')
AUDIO_DIR = os.path.join(DATA_DIR, 'audio_and_txt_files')
ANNOTATIONS_DIR = AUDIO_DIR
TRAIN_TEST_LIST = os.path.join(DATA_DIR, 'train_test.txt')
DIAGNOSIS_FILE = os.path.join(DATA_DIR, 'diagnosis.txt')
SAVED_ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'saved_assets')
MAX_SEQ_LENGTH = 500  # Max number of MFCC frames per cycle
N_MFCC = 20           # Number of MFCC features
HOP_LENGTH = 512      # For feature extraction
N_FFT = 2048          # For feature extraction
TARGET_SR = 22050     # Resample all audio to this rate

# Create saved_assets directory if it doesn't exist
os.makedirs(SAVED_ASSETS_DIR, exist_ok=True)

print(f"Data directory: {DATA_DIR}")
print(f"Audio directory: {AUDIO_DIR}")
print(f"Train/Test list: {TRAIN_TEST_LIST}")
print(f"Diagnosis file: {DIAGNOSIS_FILE}")
print(f"Saved assets directory: {SAVED_ASSETS_DIR}")

# --- Load Diagnosis Data ---
print("Loading diagnosis data...")
diagnosis_df = pd.read_csv(DIAGNOSIS_FILE, sep='\t', header=None, names=['patient_id', 'diagnosis'])
diagnosis_dict = dict(zip(diagnosis_df['patient_id'].astype(str), diagnosis_df['diagnosis']))
print(f"Loaded diagnoses for {len(diagnosis_dict)} patients")
print(f"Unique diagnoses: {diagnosis_df['diagnosis'].unique()}")

# --- 1. Load Train/Test Split Info ---
print("Loading train/test split information...")
split_df = pd.read_csv(TRAIN_TEST_LIST, sep='\t', header=None, names=['filename_base', 'split'])
train_files = set(split_df[split_df['split'] == 'train']['filename_base'])
test_files = set(split_df[split_df['split'] == 'test']['filename_base'])

print(f"Train files: {len(train_files)}")
print(f"Test files: {len(test_files)}")

# --- 2 & 3. Load Annotations, Segment Audio, Create Labels ---
all_features = []
all_labels = []
all_metadata = []  # To store associated metadata for later use
file_assignment = []  # Keep track of train/test for each segment

print("Processing audio files...")
audio_files = [f for f in os.listdir(AUDIO_DIR) if f.endswith(".wav")]
print(f"Found {len(audio_files)} audio files")

# Limit to first 50 files for testing if needed
# audio_files = audio_files[:50]

for filename in tqdm(audio_files):
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
        continue  # Skip files not in the split list

    # Parse filename for metadata
    try:
        parts = base_name.split('_')
        if len(parts) >= 5:
            patient_id, rec_index, chest_loc, acq_mode, equipment = parts[0], parts[1], parts[2], parts[3], parts[4]
            
            # Add diagnosis to metadata
            diagnosis = diagnosis_dict.get(patient_id, "Unknown")
            
            metadata = {
                'patient_id': patient_id,
                'rec_index': rec_index,
                'chest_loc': chest_loc,
                'acq_mode': acq_mode,
                'equipment': equipment,
                'filename': base_name,
                'diagnosis': diagnosis
            }
            # print(f"Metadata for {base_name}: {metadata}")
        else:
            print(f"Warning: Unexpected filename format for {base_name}")
            continue
    except Exception as e:
        print(f"Error parsing filename {base_name}: {e}")
        continue

    try:
        annotations = pd.read_csv(txt_file, sep='\t', header=None, names=['start', 'end', 'crackles', 'wheezes'])
    except Exception as e:
        print(f"Error reading annotation file {txt_file}: {e}")
        continue

    try:
        y, sr = librosa.load(wav_file, sr=TARGET_SR)  # Load and resample
    except Exception as e:
        print(f"Error loading {wav_file}: {e}")
        continue

    for index, row in annotations.iterrows():
        try:
            start_sample = int(librosa.time_to_samples(row['start'], sr=sr))
            end_sample = int(librosa.time_to_samples(row['end'], sr=sr))
            
            if start_sample >= len(y) or end_sample > len(y):
                print(f"Warning: Segment out of range for {filename} at {row['start']}-{row['end']}")
                continue
                
            segment = y[start_sample:end_sample]

            if len(segment) < sr * 0.1:  # Skip very short segments (less than 0.1s)
                continue

            # Extract Features
            mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
            mfccs = mfccs.T  # Transpose to (time_steps, n_mfcc)

            if mfccs.shape[0] > 0:  # Ensure MFCCs were extracted
                all_features.append(mfccs)
                # Multi-label: [crackles, wheezes]
                all_labels.append([int(row['crackles']), int(row['wheezes'])])
                all_metadata.append(metadata)
                file_assignment.append(current_split)
        except Exception as e:
            print(f"Error processing segment in {filename} at {row['start']}-{row['end']}: {e}")
            continue

print(f"Successfully processed {len(all_features)} respiratory cycles.")

# --- 5. Handle Variable Lengths ---
from tensorflow.keras.preprocessing.sequence import pad_sequences

print("Padding sequences...")
# Check sequence lengths
lengths = [f.shape[0] for f in all_features]
print(f"Max sequence length: {np.max(lengths)}, 95th percentile: {np.percentile(lengths, 95)}")

features_padded = []
for feature in all_features:
    # Pad or truncate
    if feature.shape[0] < MAX_SEQ_LENGTH:
        pad_width = MAX_SEQ_LENGTH - feature.shape[0]
        padded = np.pad(feature, ((0, pad_width), (0, 0)), mode='constant')
        features_padded.append(padded)
    else:
        features_padded.append(feature[:MAX_SEQ_LENGTH, :])

features_padded = np.array(features_padded, dtype='float32')
labels = np.array(all_labels, dtype='int')

# --- 6. Split Data ---
train_indices = [i for i, split in enumerate(file_assignment) if split == 'train']
test_indices = [i for i, split in enumerate(file_assignment) if split == 'test']

X_train = features_padded[train_indices]
y_train = labels[train_indices]
X_test = features_padded[test_indices]
y_test = labels[test_indices]

# Store metadata associated with test set for later analysis
metadata_test = [all_metadata[i] for i in test_indices]

print(f"Train set shape: {X_train.shape}, Labels: {y_train.shape}")
print(f"Test set shape: {X_test.shape}, Labels: {y_test.shape}")

# --- 7. Normalize Features ---
print("Normalizing features...")
# Reshape for StandardScaler (needs 2D input: samples x features)
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

# Calculate dataset statistics for the UI's charts
chest_locations = {}
equipment_types = {}
diagnoses = {}

for idx, meta in enumerate(all_metadata):
    loc = meta['chest_loc']
    equip = meta['equipment']
    diagnosis = meta['diagnosis']
    has_crackles = all_labels[idx][0]
    has_wheezes = all_labels[idx][1]
    
    # Update location stats
    if loc not in chest_locations:
        chest_locations[loc] = {'total': 0, 'crackles': 0, 'wheezes': 0}
    chest_locations[loc]['total'] += 1
    if has_crackles:
        chest_locations[loc]['crackles'] += 1
    if has_wheezes:
        chest_locations[loc]['wheezes'] += 1
    
    # Update equipment stats
    if equip not in equipment_types:
        equipment_types[equip] = {'total': 0, 'crackles': 0, 'wheezes': 0}
    equipment_types[equip]['total'] += 1
    if has_crackles:
        equipment_types[equip]['crackles'] += 1
    if has_wheezes:
        equipment_types[equip]['wheezes'] += 1
        
    # Update diagnosis stats
    if diagnosis not in diagnoses:
        diagnoses[diagnosis] = {'total': 0, 'crackles': 0, 'wheezes': 0}
    diagnoses[diagnosis]['total'] += 1
    if has_crackles:
        diagnoses[diagnosis]['crackles'] += 1
    if has_wheezes:
        diagnoses[diagnosis]['wheezes'] += 1

# Convert to prevalence rates
prevalence_by_location = {}
for loc, stats in chest_locations.items():
    if stats['total'] > 0:
        prevalence_by_location[loc] = {
            'crackles': round(stats['crackles'] / stats['total'], 2),
            'wheezes': round(stats['wheezes'] / stats['total'], 2)
        }

prevalence_by_equipment = {}
for equip, stats in equipment_types.items():
    if stats['total'] > 0:
        prevalence_by_equipment[equip] = {
            'crackles': round(stats['crackles'] / stats['total'], 2),
            'wheezes': round(stats['wheezes'] / stats['total'], 2)
        }

prevalence_by_diagnosis = {}
for diag, stats in diagnoses.items():
    if stats['total'] > 0:
        prevalence_by_diagnosis[diag] = {
            'crackles': round(stats['crackles'] / stats['total'], 2),
            'wheezes': round(stats['wheezes'] / stats['total'], 2),
            'count': stats['total']
        }

# Overall prevalence
total_segments = len(all_labels)
total_crackles = sum(1 for label in all_labels if label[0] == 1)
total_wheezes = sum(1 for label in all_labels if label[1] == 1)

overall_prevalence = {
    'crackles': round(total_crackles / total_segments, 2),
    'wheezes': round(total_wheezes / total_segments, 2)
}

# Create stats object
dataset_stats = {
    'overall_prevalence': overall_prevalence,
    'prevalence_by_location': prevalence_by_location,
    'prevalence_by_equipment': prevalence_by_equipment,
    'prevalence_by_diagnosis': prevalence_by_diagnosis
}

# Create a mapping from patient_id to diagnosis for quick lookup
patient_diagnosis_map = {}
for patient_id, diagnosis in diagnosis_dict.items():
    patient_diagnosis_map[patient_id] = diagnosis

# Save preprocessed data and scaler
print("Saving preprocessed data and scaler...")
np.savez(os.path.join(SAVED_ASSETS_DIR, 'preprocessed_data.npz'), 
         X_train=X_train_scaled, y_train=y_train, 
         X_test=X_test_scaled, y_test=y_test)
joblib.dump(scaler, os.path.join(SAVED_ASSETS_DIR, 'scaler.joblib'))
joblib.dump(dataset_stats, os.path.join(SAVED_ASSETS_DIR, 'dataset_stats.joblib'))
joblib.dump(patient_diagnosis_map, os.path.join(SAVED_ASSETS_DIR, 'patient_diagnosis_map.joblib'))

print("Data preparation complete.")
print(f"Files saved to {SAVED_ASSETS_DIR}") 