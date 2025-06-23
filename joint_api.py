import os
import math
import torch
import torch.nn.functional as F
import torchaudio
import librosa
import numpy as np
from io import BytesIO
from flask import Flask, request, jsonify
from google.cloud import storage
from model import MambaHead, SSLModel, EmotionClassifier

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

# You can set these environment variables before running, or hardcode paths here:
#   - MAMBA_GCS_BUCKET / MAMBA_GCS_BLOB for the Mamba model .pth
#   - SER_GCS_BUCKET / SER_GCS_BLOB for the SER model .pth
#   - XLSR_GCS_BUCKET / XLSR_GCS_BLOB for the XLSR checkpoint

MODEL_BUCKET    = 'audio-classifier-mamba'
MAMBA_GCS_BLOB     = 'pretrained_weights/best_2_a3.pth'
SER_GCS_BLOB       = 'pretrained_weights/81_trans.pth'
XLSR_GCS_BLOB      = 'pretrained_weights/xlsr2_300m.pt'

# Local filenames for downloaded weights
MAMBA_LOCAL_PATH   = 'best_2_a3.pth'
SER_LOCAL_PATH     = '81_trans.pth'
XLSR_LOCAL_PATH    = 'xlsr2_300m.pt'

# If GCS paths are provided, download them at startup
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

# Audio config (can be overridden via environment variables)
TARGET_SR       = int(os.environ.get('TARGET_SR', 16000))
CHUNK_LEN_S     = float(os.environ.get('CHUNK_LEN_S', 2.0))  # chunk length in seconds
NUM_SER_CLASSES = int(os.environ.get('NUM_SER_CLASSES', 6))

# Emotion labels mapping
EMO_CLASSES = {
    0: 'Angry', 1: 'Disgusted', 2: 'Fearful',
    3: 'Happy', 4: 'Neutral',    5: 'Sad'
}

# Device and model initialization
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Instantiate and load SSL backbone (uses downloaded XLSR checkpoint)
ssl = SSLModel(device)
ssl.model.eval().to(device)

# Instantiate and load Mamba head
#   We assume that MambaHead’s constructor takes (args, device),
#   but here we’ll create a dummy 'args' namespace for the relevant hyperparameters.
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
mamba_head = MambaHead(dummy_args, device)
mamba_head.load_state_dict(torch.load(MAMBA_LOCAL_PATH, map_location=device))
mamba_head.eval().to(device)

# Instantiate and load SER head
ser_head = EmotionClassifier(ssl, feat_dim=ssl.out_dim, num_classes=NUM_SER_CLASSES)
ser_head.load_state_dict(torch.load(SER_LOCAL_PATH, map_location=device))
ser_head.eval().to(device)


# ----------------------------
# Prediction endpoint
# ----------------------------
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """
    Expects JSON with:
    {
      "instances": [
        { "audio_path": "gs://bucket/path/to/audio.wav" }
      ]
    }
    Downloads the audio from GCS, performs chunk-wise inference for emotion + deepfake detection,
    and returns a JSON with per-chunk predictions.
    """
    if request.method == 'GET':
        return "", 200

    data = request.get_json(silent=True)
    if not data or 'instances' not in data:
        return jsonify({'error': 'Please provide instances'}), 400

    instances = data['instances']
    inst = instances[-1]  # only process the last instance
    if 'audio_path' not in inst:
        return jsonify({'error': 'Please provide audio_path parameter'}), 400

    audio_path = inst['audio_path']
    if not audio_path.startswith('gs://'):
        return jsonify({'error': 'audio_path must be a gs:// URI'}), 400

    try:
        # 1) Download the audio to a temporary file
        bucket, blob = parse_gcs_path(audio_path)
        local_audio = 'temp_audio.wav'
        download_from_gcs(bucket, blob, local_audio)

        # 2) Load waveform with librosa (mono, TARGET_SR)
        waveform, sr = librosa.load(local_audio, sr=TARGET_SR)
        total_samples = waveform.shape[0]
        segment_length = int(CHUNK_LEN_S * TARGET_SR)
        num_segments = int(math.ceil(total_samples / segment_length))

        predictions = []

        def classify_sigmoid_output(score):
            if score < 0.10:
                return "Fake"
            elif score < 0.25:
                return "Suspicious"
            elif score < 0.50:
                return "Slightly Suspicious"
            else:
                return "Real / Confident"

        for i in range(1):
            start = i * segment_length
            end = min((i + 1) * segment_length, total_samples)
            if end - start < segment_length:
                # If last segment is shorter, pad with zeros to exactly segment_length
                pad_size = segment_length - (end - start)
                segment = np.concatenate([waveform[start:end], np.zeros(pad_size, dtype=waveform.dtype)])
            else:
                segment = waveform[start:end]

            # Convert to tensor and send to device
            segment_tensor = torch.tensor(segment).unsqueeze(0).to(device)

            with torch.no_grad():
                # 1) Extract features via SSL backbone
                final_emb, all_layers = ssl.extract_feat(segment_tensor)
                # 2) Deepfake detection (Mamba head)
                deepfake_out = mamba_head(final_emb)
                deepfake_probs = torch.softmax(deepfake_out, dim=1)
                deepfake_score = deepfake_probs[0, 1].item()
                deepfake_label = classify_sigmoid_output(deepfake_score)

                # 3) Emotion recognition (SER head)
                layer_feats = all_layers[0] if isinstance(all_layers, tuple) else all_layers
                raw = layer_feats[10]
                if isinstance(raw, tuple):
                    raw = raw[0]
                feat = raw.squeeze(1)              # (time_steps, feat_dim)
                emb_batch = feat.unsqueeze(0)      # (1, time_steps, feat_dim)
                lengths = [feat.size(0)]

                logits, _ = ser_head(emb_batch, lengths)
                emo_probs = logits.softmax(dim=1)
                emo_idx = logits.argmax(dim=1).item()
                emo_score = emo_probs.max().item()
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
                
            })

        os.remove(local_audio)
        return jsonify({'predictions': predictions})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ----------------------------
# Run the Flask app
# ----------------------------
if __name__ == '__main__':
    # Use host=0.0.0.0 and port=8080 for compatibility with many deployment environments
    app.run(host='0.0.0.0', port=8080, debug=False)

