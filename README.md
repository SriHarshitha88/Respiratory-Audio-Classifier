# Abnormal Respiratory Sound Detector

This project detects the presence of crackles and wheezes in respiratory audio recordings using an LSTM model. It includes data preparation, model training, a Flask backend API, and a web-based frontend UI for audio visualization and analysis.

## Features

- Audio upload and waveform visualization
- LSTM-based analysis of respiratory sounds
- Detection of crackles and wheezes in audio segments
- Visual highlighting of abnormal segments on the waveform
- Dataset insights showing prevalence of abnormalities by chest location and recording equipment

## Project Structure

```
├── data/
│   └── raw/
│       ├── audio_and_txt_files/ (symlink to ICBHI_final_database)
│       └── train_test.txt
├── saved_assets/
│   ├── preprocessed_data.npz
│   ├── scaler.joblib
│   ├── dataset_stats.joblib
│   └── best_model.keras
├── scripts/
│   ├── prepare_data.py
│   └── train_model.py
├── static/
│   └── css/
│       └── style.css
├── templates/
│   └── index.html
├── app.py
└── requirements.txt
```

## Setup and Installation

1. Clone the repository:
   ```bash
   git clone <your-repository-url>
   cd abnormal-respiratory-sound-detector
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Ensure the dataset is properly linked:
   - The ICBHI respiratory sound database should be available at `ICBHI_final_database`
   - There should be a symlink from `data/raw/audio_and_txt_files` to the database directory
   - The `train_test.txt` file should be in `data/raw/`

## Running the Project

1. Prepare the data (this may take some time):
   ```bash
   python scripts/prepare_data.py
   ```

2. Train the model (requires prepared data):
   ```bash
   python scripts/train_model.py
   ```

3. Run the Flask application:
   ```bash
   python app.py
   ```

4. Open your web browser and navigate to `http://127.0.0.1:5000/`

## Usage

1. Click the "Choose File" button to select a respiratory audio recording (WAV format).
2. Once the file is loaded and the waveform is displayed, click the "Analyze Audio" button.
3. Wait for processing to complete (indicated by the "Processing..." message).
4. View the prediction results below the waveform - segments containing abnormal sounds will be highlighted directly on the waveform.
5. Use the Play/Pause button to listen to the audio.
6. Explore the dataset insights charts at the bottom of the page to understand the prevalence of abnormalities in the training data.

## Notes

- The model performance depends on the quality and quantity of the training data.
- Processing large audio files may take some time.
- For optimal results, use high-quality respiratory recordings with minimal background noise. 