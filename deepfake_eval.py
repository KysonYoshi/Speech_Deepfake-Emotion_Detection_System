import argparse
import os
import glob
import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
from model import Model  # Make sure model.py is in the same directory, and the Model constructor matches training

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc

def predict_file(filepath, device, model):
    """
    Load the audio file, split it into 3-second segments,
    perform inference on each segment,
    and average the predicted scores (probability of class 1)
    to determine the final prediction for the audio file:
        avg_score > 0.5 -> predict 1 (fake)
        otherwise predict 0 (real)
    """
    waveform, sample_rate = librosa.load(filepath, sr=16000)
    segment_length = 4 * sample_rate  # Number of samples in a 1-second segment (16000 for 16000 Hz)
    num_samples = waveform.shape[0]
    num_segments = int(np.ceil(num_samples / segment_length))
    scores = []
    
    for i in range(num_segments):
        start = i * segment_length
        end = (i + 1) * segment_length
        segment = waveform[start:end]
        # If the last segment is shorter than 3 seconds, pad with zeros
        if end > num_samples:
            break
        segment_tensor = torch.tensor(segment).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(segment_tensor)
            prob = torch.softmax(output, dim=1)
            score = prob[0, 1].item()
        scores.append(score)

    plt.figure(figsize=(8, 4))
    plt.plot(scores, marker='o', linestyle='-', color='blue')
    plt.title("Deepfake Detection Scores")
    plt.xlabel("Sample Index")
    plt.ylabel("Deepfake Score (Lower = More Fake)")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig("deepfake_scores_plot.png", dpi=300)
    plt.close()
    
    avg_score = sum(scores) / len(scores)
    final_pred = 1 if avg_score > 0.4997 else 0
    return final_pred, avg_score

if __name__ == '__main__':
    default_data_folder_path = "/home/cl6933/XLSR-Mamba/audio"
    default_model_path = "/home/cl6933/XLSR-Mamba/model/best_2_a3.pth"

    parser = argparse.ArgumentParser(description='Run inference on all real and fake .wav files in the directory and compute evaluation metrics.')
    parser.add_argument('--data_path', type=str, default=default_data_folder_path, help='Path to folder containing "real" and "fake" subdirectories with audio files')
    parser.add_argument('--model_path', type=str, default=default_model_path, help='Path to model state dict (e.g., best.pth)')
    # The following parameters must match those used during training; modify if necessary
    parser.add_argument('--emb-size', type=int, default=144, metavar='N', help='Embedding size of the model')
    parser.add_argument('--num_encoders', type=int, default=12, metavar='N', help='Number of encoders in Mamba blocks')
    parser.add_argument('--FT_W2V', default=True, type=lambda x: (str(x).lower() in ['true', 'yes', '1']), help='Whether to fine-tune W2V')
    parser.add_argument('--algo', type=int, default=3, help='RawBoost algorithm setting (e.g., 3 for DF, 5 for LA and ITW)')
    parser.add_argument('--loss', type=str, default='WCE', help='Loss function type')
    args = parser.parse_args()

    # Set device (GPU or CPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create model (parameters must match training settings)
    model = Model(args, device)
    model = model.to(device)
    # Load model weights
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded model: {args.model_path}")

    # Gather all .wav files under real and fake subdirectories
    real_files = glob.glob(os.path.join(args.data_path, "real", "*.wav"))
    fake_files = glob.glob(os.path.join(args.data_path, "fake", "*.wav"))
    print(f"Found {len(real_files)} real files and {len(fake_files)} fake files.")

    # Store ground truth, predictions, average scores, and file paths for all files
    y_true = []
    y_pred = []
    y_score = []
    file_paths = []

    def classify_sigmoid_output(score):
        if score < 0.10:
            return "Fake"
        elif score < 0.25:
            return "Suspicious"
        elif score < 0.5:
            return "Slightly Suspicious"
        else:
            return "Real / Confident"

    # Run inference on real files (assume label = 1)
    for filepath in real_files:
        file_paths.append(filepath)
        pred, avg_score = predict_file(filepath, device, model)
        y_true.append(1)
        y_pred.append(pred)
        y_score.append(avg_score)
        print(f"[REAL] {filepath} -> Predicted: {pred}, Avg Score: {avg_score:.4f}, {classify_sigmoid_output(avg_score)}")

    # Run inference on fake files (assume label = 0)
    for filepath in fake_files:
        file_paths.append(filepath)
        pred, avg_score = predict_file(filepath, device, model)
        y_true.append(0)
        y_pred.append(pred)
        y_score.append(avg_score)
        print(f"[FAKE] {filepath} -> Predicted: {pred}, Avg Score: {avg_score:.4f}, {classify_sigmoid_output(avg_score)}")

    # Compute evaluation metrics
    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    accuracies = [(threshold, accuracy_score(y_true, [1 if s > threshold else 0 for s in y_score]))
                  for threshold in thresholds]
    best_threshold, best_accuracy = max(accuracies, key=lambda x: x[1])

    print(f"Best threshold for highest accuracy: {best_threshold:.4f}")

    print("\nEvaluation Metrics:")
    print("Confusion Matrix:")
    print(cm)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")

    # Plot and save ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig("roc_curve.png")

    # Print Misclassified Files
    print("\nMisclassified Files:")
    for i, fp in enumerate(file_paths):
        if y_true[i] != y_pred[i]:
            print(f"File: {fp} | Ground Truth: {y_true[i]} | Predicted: {y_pred[i]} | Avg Score: {y_score[i]:.4f}")
