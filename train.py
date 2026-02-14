import os
import glob
import numpy as np
import librosa
import joblib
from sklearn.ensemble import RandomForestClassifier

# --- Settings ---
SR = 16000
N_MFCC = 40
MAX_LEN = 5.0

def extract_features(path):
    # Load and standardize audio length
    y, _ = librosa.load(path, sr=SR, mono=True)
    max_samples = int(SR * MAX_LEN)
    y = y[:max_samples] if len(y) > max_samples else np.pad(y, (0, max_samples - len(y)))
    
    # Extract MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC)
    mfcc_norm = (mfcc - np.mean(mfcc, axis=1, keepdims=True)) / (np.std(mfcc, axis=1, keepdims=True) + 1e-9)
    
    # Aggregate to 80 features (Mean + Std)
    return np.concatenate([np.mean(mfcc_norm, axis=1), np.std(mfcc_norm, axis=1)])

def run_training():
    X, y = [], []
    # Adjust these folder names if necessary
    categories = {'vish': 0, 'normal': 1}
    
    for folder, label in categories.items():
        if not os.path.exists(folder):
            print(f"❌ Error: Folder '{folder}' not found!")
            continue
            
        files = glob.glob(os.path.join(folder, "*.mp3"))
        print(f"Reading {len(files)} files from {folder}...")
        
        for f in files:
            try:
                X.append(extract_features(f))
                y.append(label)
            except Exception as e:
                print(f"Skipping {f}: {e}")

    if len(X) > 0:
        model = RandomForestClassifier(n_estimators=300, random_state=42)
        model.fit(X, y)
        joblib.dump(model, 'vishing_model.pkl')
        print("✅ Success! 'vishing_model.pkl' created.")
    else:
        print("❌ Training failed: No audio files were processed.")

if __name__ == "__main__":
    run_training()