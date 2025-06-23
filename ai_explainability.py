#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import torch
import torch.nn.functional as F
import json
import librosa
import numpy as np
import pandas as pd
from io import BytesIO
from flask import Flask, request, jsonify
from google.cloud import storage

# AIX360 explainability imports
from aix360.algorithms.tsutils.tsframe import tsFrame
from aix360.algorithms.tssaliency.tssaliency import TSSaliencyExplainer

from model import Model, SSLModel, EmotionClassifier

# ----------------------------
# GCS helper functions
# ----------------------------

def download_from_gcs(bucket_name, blob_name, destination_path):
    """Download a blob from a specified GCS bucket to destination_path"""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(destination_path)
    print(f"Downloaded gs://{bucket_name}/{blob_name} to {destination_path}")

def parse_gcs_path(gcs_path):
    """Parse a GCS path (e.g., gs://bucket-name/path/to/file) into (bucket, blob)"""
    if not gcs_path.startswith("gs://"):
        raise ValueError("GCS path must start with gs://")
    parts = gcs_path[5:].split("/", 1)
    if len(parts) != 2:
        raise ValueError("Invalid gs:// path format")
    return parts[0], parts[1]

# ----------------------------
# Configuration & Model Download
# ----------------------------

MODEL_BUCKET    = 'audio-classifier-mamba'
MAMBA_GCS_BLOB  = 'pretrained_weights/best_2_a3.pth'
SER_GCS_BLOB    = 'pretrained_weights/81_trans.pth'
XLSR_GCS_BLOB   = 'pretrained_weights/xlsr2_300m.pt'

MAMBA_LOCAL_PATH = 'best_2_a3.pth'
SER_LOCAL_PATH   = '81_trans.pth'
XLSR_LOCAL_PATH  = 'xlsr2_300m.pt'

if MAMBA_GCS_BLOB:
    print("Downloading Mamba model weights from GCS...")
    download_from_gcs(MODEL_BUCKET, MAMBA_GCS_BLOB, MAMBA_LOCAL_PATH)

if SER_GCS_BLOB:
    print("Downloading SER model weights from GCS...")
    download_from_gcs(MODEL_BUCKET, SER_GCS_BLOB, SER_LOCAL_PATH)

if XLSR_GCS_BLOB:
    print("Downloading XLSR checkpoint from GCS...")
    download_from_gcs(MODEL_BUCKET, XLSR_GCS_BLOB, XLSR_LOCAL_PATH)

# ----------------------------
# Flask app setup
# ----------------------------
app = Flask(__name__)

TARGET_SR       = int(os.environ.get('TARGET_SR', 16000))
CHUNK_LEN_S     = float(os.environ.get('CHUNK_LEN_S', 4.0))
NUM_SER_CLASSES = int(os.environ.get('NUM_SER_CLASSES', 6))

EMO_CLASSES = {
    0: 'Angry', 1: 'Disgusted', 2: 'Fearful',
    3: 'Happy', 4: 'Neutral',    5: 'Sad'
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------------------------
# Load models
# ----------------------------
ssl = SSLModel(device)
ssl.model.eval().to(device)

class DummyArgs:
    def __init__(self):
        self.emb_size = int(os.environ.get('MAMBA_EMB_SIZE', 144))
        self.num_encoders = int(os.environ.get('MAMBA_NUM_ENCODERS', 12))
        self.FT_W2V = bool(int(os.environ.get('MAMBA_FT_W2V', 1)))
        self.algo = int(os.environ.get('MAMBA_ALGO', 3))
        self.loss = os.environ.get('MAMBA_LOSS', 'WCE')
        self.target_sr = TARGET_SR
        self.num_ser_classes = NUM_SER_CLASSES

dummy_args = DummyArgs()
mamba_head = Model(dummy_args, device)
mamba_head.load_state_dict(torch.load(MAMBA_LOCAL_PATH, map_location=device))
mamba_head.eval().to(device)

ser_head = EmotionClassifier(ssl, feat_dim=ssl.out_dim, num_classes=NUM_SER_CLASSES)
ser_head.load_state_dict(torch.load(SER_LOCAL_PATH, map_location=device))
ser_head.eval().to(device)

# ----------------------------
# Explainability setup
# ----------------------------

def simple_mc_gradient(x: np.ndarray, fn, n_samples: int = 25, mu: float = 1e-4):
    """
    Batched Monte Carlo gradient estimator.
    x: (L,1)          → waveform column vector
    fn: (B, L) -> (B,) → deepfake_model_numpy
    returns: (L,1)    → saliency gradient
    """
    # 1) reshape to (1, L)
    x0 = x.T.astype(np.float32)            # (1, L)

    # 2) sample all perturbations at once
    u = np.random.randn(n_samples, x0.shape[1]).astype(np.float32)  # (S, L)
    xs = x0 + mu * u                                               # (S, L)

    # 3) one call on the clean signal, one on the whole batch
    f_x  = fn(x0)    # → shape (1,)
    f_xs = fn(xs)    # → shape (S,)

    # 4) monte-carlo gradient
    #    df: (S,1)      = (f_xs - f_x)[:,None]
    #    g : (L,)       = mean(df * u, axis=0) / mu
    df = (f_xs - f_x)[:, None]
    g  = (df * u).mean(axis=0) / mu  # shape (L,)

    # 5) return as column vector (L,1)
    return g[:, None]

def deepfake_model_numpy(batch_x: np.ndarray) -> np.ndarray:
    """
    Wrapper to call the SSL + Mamba head on numpy waveform inputs.
    batch_x: (B, L)
    returns: (B,) deepfake 'realness' score = probability of class 1
    """
    xs = torch.from_numpy(batch_x.astype(np.float32)).to(device)  # (B,L)
    with torch.no_grad():
        out = mamba_head(xs)                               # (B,2)
        probs = F.softmax(out, dim=1)                             # (B,2)
        scores = probs[:, 1].cpu().numpy()                        # (B,)
    return scores

# Segment length in samples
segment_length = int(CHUNK_LEN_S * TARGET_SR)

# Instantiate the TSSaliencyExplainer
explainer = TSSaliencyExplainer(
    model=deepfake_model_numpy,
    input_length=segment_length,
    feature_names=["waveform"],
    n_samples=25,
    gradient_samples=10,
    gradient_function=simple_mc_gradient,
    random_seed=42,
)

# ----------------------------
# Prediction endpoint
# ----------------------------
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return "", 200

    data = request.get_json(silent=True)
    if not data or 'instances' not in data:
        return jsonify({'error': 'Please provide instances'}), 400

    inst = data['instances'][-1]
    audio_path = inst.get('audio_path')
    if not audio_path or not audio_path.startswith('gs://'):
        return jsonify({'error': 'audio_path must be a gs:// URI'}), 400

    try:
        # Download and load audio
        bucket, blob = parse_gcs_path(audio_path)
        local_audio = 'temp_audio.wav'
        download_from_gcs(bucket, blob, local_audio)

        waveform, sr = librosa.load(local_audio, sr=TARGET_SR)
        total_samples = waveform.shape[0]
        num_segments = int(math.ceil(total_samples / segment_length))

        predictions = []
        
        for i in range(1):
            start = i * segment_length
            end = min((i + 1) * segment_length, total_samples)
            if end - start < segment_length:
                # If last segment is shorter, pad with zeros to exactly segment_length
                pad_size = segment_length - (end - start)
                segment = np.concatenate([waveform[start:end], np.zeros(pad_size, dtype=waveform.dtype)])
            else:
                segment = waveform[start:end]

       
            # Prepare timestamped DataFrame for explainer
            df_seg = pd.DataFrame({
                "timestamp": np.arange(segment_length, dtype=np.int64),
                "waveform": segment.astype(np.float32)
            })
            ts_seg = tsFrame(df_seg, timestamp_column="timestamp", columns=["waveform"], dt=1.0 / TARGET_SR)

            # Compute saliency map
            explanation = explainer.explain_instance(ts_seg)
            saliency = explanation["saliency"].squeeze().tolist()

            # Convert segment to tensor for inference
            segment_tensor = torch.from_numpy(segment).unsqueeze(0).to(device)  # (1, L)

            with torch.no_grad():
                # Deepfake detection
                deepfake_out = mamba_head(segment_tensor)
                deepfake_probs = torch.softmax(deepfake_out, dim=1)
                deepfake_score = deepfake_probs[0, 1].item()
                if deepfake_score < 0.10:
                    deepfake_label = "Fake"
                elif deepfake_score < 0.25:
                    deepfake_label = "Suspicious"
                elif deepfake_score < 0.50:
                    deepfake_label = "Slightly Suspicious"
                else:
                    deepfake_label = "Real / Confident"

                # Emotion recognition
                _ , all_layers = ssl.extract_feat(segment_tensor)
                layer_feats = all_layers[0] if isinstance(all_layers, tuple) else all_layers
                raw = layer_feats[10]
                if isinstance(raw, tuple):
                    raw = raw[0]
                feat = raw.squeeze(1)
                emb_batch = feat.unsqueeze(0)
                lengths = [feat.size(0)]
                logits, _ = ser_head(emb_batch, lengths)
                emo_probs = logits.softmax(dim=1)
                emo_idx = int(logits.argmax(dim=1).item())
                emo_score = float(emo_probs.max().item())
                emo_label = EMO_CLASSES.get(emo_idx, "Unknown")

            n_fft = 2048
            hop_length = 512
            segment_np = np.asarray(segment, dtype=np.float32)
            # Spectrogram
            S = np.abs(librosa.stft(segment_np, n_fft=n_fft, hop_length=hop_length))
            db_S = librosa.amplitude_to_db(S, ref=np.max)

            # Spectral Entropy Variation
            power = S**2
            power_norm = power / np.sum(power, axis=0, keepdims=True)
            entropy = -np.sum(power_norm * np.log2(power_norm + 1e-12), axis=0)
            times = librosa.frames_to_time(np.arange(entropy.size), sr=sr, hop_length=hop_length)

            # Pitch Variation
            f0 = librosa.yin(segment_np,
                            fmin=librosa.note_to_hz('C2'),
                            fmax=librosa.note_to_hz('C7'),
                            sr=sr,
                            frame_length=n_fft,
                            hop_length=hop_length)
            
            # Silence Ratio and SNR
            intervals = librosa.effects.split(segment_np, top_db=20)
            mask = np.zeros_like(segment_np, dtype=bool)
            for start, end in intervals:
                mask[start:end] = True

            signal_power = np.mean(segment_np[mask]**2) if np.any(mask) else 0
            noise_power = np.mean(segment_np[~mask]**2) if np.any(~mask) else 1e-12
            snr = 10 * np.log10(signal_power / noise_power)
            silence_ratio = np.sum(~mask) / len(segment_np)

            # Spectral Centroid (Hz)
            centroid = librosa.feature.spectral_centroid(y=segment_np, sr=sr, hop_length=hop_length)[0]

            # Spectral Bandwidth
            bandwidth = librosa.feature.spectral_bandwidth(y=segment_np, sr=sr, hop_length=hop_length)[0]

            # Spectral Rolloff (Hz)
            rolloff = librosa.feature.spectral_rolloff(y=segment_np, sr=sr, roll_percent=0.85, hop_length=hop_length)[0]

            # Spectral Flatness
            flatness = librosa.feature.spectral_flatness(y=segment_np, hop_length=hop_length)[0]

            # Zero Crossing Rate
            zcr = librosa.feature.zero_crossing_rate(y=segment_np, hop_length=hop_length)[0]

            # RMS Energy
            rms = librosa.feature.rms(y=segment_np, hop_length=hop_length)[0]

            predictions.append({
                'chunk_index': i + 1,
                'emotion': emo_label,
                'emotion_score': round(emo_score, 4),
                'deepfake_label': deepfake_label,
                'deepfake_score': round(deepfake_score, 4),
                'spectrogram': db_S.tolist(),
                'spectral_entropy': entropy.tolist(),
                'pitch_variation': f0.tolist(),
                'snr': round(snr.item(), 2),
                'silence_ratio': round(silence_ratio.item(), 2),
                'spectral_centroid': centroid.tolist(),
                'spectral_bandwidth': bandwidth.tolist(),
                'spectral_rolloff': rolloff.tolist(),
                'spectral_flatness': flatness.tolist(),
                'zero_crossing_rate': zcr.tolist(),
                'rms_energy': rms.tolist(),
                'time_line': times.tolist(),
                'saliency': saliency,
            })

        os.remove(local_audio)
        output = {'predictions': predictions}
        with open('predictions.json', 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        return jsonify(output), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ----------------------------
# Run the Flask app
# ----------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)

