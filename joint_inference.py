import argparse
import os
import glob
import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
from model import MambaHead, SSLModel, EmotionClassifier  # Make sure model.py is in the same directory, and the Model constructor matches training

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc

def predict_file(filepath, device, ssl, mamba_head, ser_head):
    """
    Load the audio file, split it into 3-second segments,
    perform inference on each segment,
    and average the predicted scores (probability of class 1)
    to determine the final prediction for the audio file:
        avg_score > 0.5 -> predict 1 (fake)
        otherwise predict 0 (real)
    """
    waveform, sample_rate = librosa.load(filepath, sr=16000)

    
    """waveform_tensor = torch.tensor(waveform).unsqueeze(0).to(device)

    
    with torch.no_grad():
        output = model(waveform_tensor)
        prob = torch.softmax(output, dim=1)
        avg_score = prob[0, 1].item()"""
    segment_length = 2 * sample_rate  # Number of samples in a 1-second segment (16000 for 16000 Hz)
    num_samples = waveform.shape[0]
    num_segments = int(np.ceil(num_samples / segment_length))
    scores = []

    ser_classes = {0: 'Angry',
                  1: 'Disgusted',
                  2: 'Fearful',
                  3: 'Happy',
                  4: 'Neutral',
                  5: 'Sad'}
    def classify_sigmoid_output(score):
        if score < 0.10:
            return "Fake"
        elif score < 0.25:
            return "Suspicious"
        elif score < 0.5:
            return "Slightly Suspicious"
        else:
            return "Real / Confident"
    
    for i in range(num_segments):
        start = i * segment_length
        end = (i + 1) * segment_length
        segment = waveform[start:end]
        # If the last segment is shorter than 3 seconds, pad with zeros
        if end > num_samples:
            break
        segment_tensor = torch.tensor(segment).unsqueeze(0).to(device)
        with torch.no_grad():
            final_emb, all_layers = ssl.extract_feat(segment_tensor)
            deepfake_out = mamba_head(final_emb)
            layer_feats = all_layers[0] if isinstance(all_layers, tuple) else all_layers
            raw = layer_feats[10]
            if isinstance(raw, tuple):
                raw = raw[0]
            feat = raw.squeeze(1)            # (time, feat_dim)
            emb_batch = feat.unsqueeze(0)     # (1, time, feat_dim)
            lengths = [feat.size(0)]

            # classification
            logits, cls_repr = ser_head(emb_batch, lengths)
            emo_logits = logits.softmax(dim=1)  # (1, num_classes)
            emo_idx = logits.argmax(dim=1).item()
            emo_score = logits.softmax(dim=1).max().item()
            deepfake_out_prob = torch.softmax(deepfake_out, dim=1)
            deepfake_out_prob_score = deepfake_out_prob[0, 1].item()
        start_s = start / args.target_sr
        end_s = min(end, num_samples) / args.target_sr
        print(f"chunk {i+1}/{num_segments} [{start_s:.2f}s–{end_s:.2f}s] -> Predicted emotion index: {ser_classes[emo_idx]}, Score: {emo_score:.4f}\nDeepfake detect result：{classify_sigmoid_output(deepfake_out_prob_score)} (Confident Score：{deepfake_out_prob_score:.4f})")



if __name__ == '__main__':
    default_data_folder_path = "/home/cl6933/XLSR-Mamba/release_in_the_wild/real/314.wav"
    mamba_model_path = "/home/cl6933/XLSR-Mamba/model/best_2_a3.pth"

    parser = argparse.ArgumentParser(description='Run inference on all real and fake .wav files in the directory and compute evaluation metrics.')
    parser.add_argument('--data_path', type=str, default=default_data_folder_path, help='Path to folder containing "real" and "fake" subdirectories with audio files')
    parser.add_argument('--mamba_model_path', type=str, default=mamba_model_path, help='Path to model state dict (e.g., best.pth)')
    parser.add_argument('--ser_model_path',
                        default='/home/cl6933/XLSR-Mamba/model/81_trans.pth',
                        help='Path to the trained model checkpoint')
    # The following parameters must match those used during training; modify if necessary
    parser.add_argument('--emb-size', type=int, default=144, metavar='N', help='Embedding size of the model')
    parser.add_argument('--num_encoders', type=int, default=12, metavar='N', help='Number of encoders in Mamba blocks')
    parser.add_argument('--FT_W2V', default=True, type=lambda x: (str(x).lower() in ['true', 'yes', '1']), help='Whether to fine-tune W2V')
    parser.add_argument('--algo', type=int, default=3, help='RawBoost algorithm setting (e.g., 3 for DF, 5 for LA and ITW)')
    parser.add_argument('--loss', type=str, default='WCE', help='Loss function type')
    parser.add_argument('--target_sr',
                        type=int,
                        default=16000,
                        help='Target sampling rate')
    parser.add_argument('--num_ser_classes',
                        type=int,
                        default=6,
                        help='Number of emotion classes')
    args = parser.parse_args()

    # Set device (GPU or CPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create model (parameters must match training settings)
    mamba_head = MambaHead(args, device)
    mamba_head = mamba_head.to(device)
    ssl = SSLModel(device)
    ssl.model.eval().to(device)
    ser_head = EmotionClassifier(ssl, feat_dim=ssl.out_dim, num_classes=args.num_ser_classes)
    ser_head.load_state_dict(torch.load(args.ser_model_path, map_location=device))
    ser_head.eval().to(device)
    
    # Load model weights
    mamba_head.load_state_dict(torch.load(args.mamba_model_path, map_location=device))
    mamba_head.eval()
    print(f"Loaded model: {args.mamba_model_path}")

    predict_file(args.data_path, device, ssl, mamba_head, ser_head)
