import os
import math
import argparse

import torch
import torch.nn.functional as F
import torchaudio

from model import SSLModel, EmotionClassifier

def predict(args):
    # Set up device and models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ssl = SSLModel(device)
    ssl.model.eval().to(device)

    model = EmotionClassifier(ssl, feat_dim=ssl.out_dim, num_classes=args.num_ser_classes)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval().to(device)

    # Gather input paths
    if os.path.isdir(args.input_path):
        paths = [
            os.path.join(args.input_path, fn)
            for fn in os.listdir(args.input_path)
            if fn.lower().endswith('.wav')
        ]
    else:
        paths = [args.input_path]

    chunk_len_s = 2.0
    chunk_size = int(args.target_sr * chunk_len_s)

    for path in paths:
        if not os.path.isfile(path):
            print(f"[WARN] File not found: {path}")
            continue

        # Load & preprocess
        wav, sr = torchaudio.load(path)                  # (channels, samples)
        wav = wav.mean(dim=0)                             # mono
        if sr != args.target_sr:
            wav = torchaudio.functional.resample(wav, sr, args.target_sr)
        wav = wav.to(device)
        print(wav.max(), wav.min(), wav.size())

        total_samples = wav.size(0)
        num_chunks = math.ceil(total_samples / chunk_size)
        base = os.path.basename(path)

        classes = {0: 'Angry',
                  1: 'Disgusted',
                  2: 'Fearful',
                  3: 'Happy',
                  4: 'Neutral',
                  5: 'Sad'}

        for i in range(num_chunks):
            start_sample = i * chunk_size
            end_sample = start_sample + chunk_size
            chunk = wav[start_sample:end_sample]

            # pad last chunk if needed
            if chunk.size(0) < chunk_size:
                break

            # SSL feature extraction
            with torch.no_grad():
                # model expects shape (batch, time)
                input_batch = chunk.unsqueeze(0)  # (1, chunk_size)
                final_emb, all_layers = ssl.extract_feat(input_batch)
                layer_feats = all_layers[0] if isinstance(all_layers, tuple) else all_layers
                raw = layer_feats[10]
                if isinstance(raw, tuple):
                    raw = raw[0]
                feat = raw.squeeze(1)            # (time, feat_dim)
                emb_batch = feat.unsqueeze(0)     # (1, time, feat_dim)
                lengths = [feat.size(0)]

                # classification
                logits, cls_repr = model(emb_batch, lengths)
                pred_logits = logits.softmax(dim=1)  # (1, num_classes)
                pred_idx = logits.argmax(dim=1).item()
                pred_score = logits.softmax(dim=1).max().item()

            # compute times
            start_s = start_sample / args.target_sr
            end_s = min(end_sample, total_samples) / args.target_sr
            print(f"{base} — chunk {i+1}/{num_chunks} [{start_s:.2f}s–{end_s:.2f}s] -> Predicted emotion index: {classes[pred_idx]}, Score: {pred_score:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict emotion on 2s chunks of WAV file(s)')
    parser.add_argument('--input_path',
                        default='/home/cl6933/XLSR-Mamba/audio/real/20369.wav',
                        help='Path to a .wav file or directory of .wav files')
    parser.add_argument('--model_path',
                        default='/home/cl6933/XLSR-Mamba/model/81_trans.pth',
                        help='Path to the trained model checkpoint')
    parser.add_argument('--target_sr',
                        type=int,
                        default=16000,
                        help='Target sampling rate')
    parser.add_argument('--num_ser_classes',
                        type=int,
                        default=6,
                        help='Number of emotion classes')
    args = parser.parse_args()
    predict(args)

