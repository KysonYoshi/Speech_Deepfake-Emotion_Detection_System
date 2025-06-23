import os
import argparse
import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from model import SSLModel, EmotionClassifier
from sklearn.metrics import classification_report, confusion_matrix
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn.functional as F

# -------- Default paths --------
DEFAULT_AUDIO_DIR = '/home/cl6933/crema-d-mirror/AudioWAV'
DEFAULT_LABEL_FILE = '/home/cl6933/crema-d-mirror/finishedResponses.csv'


def collate_fn_train(batch):
    embs, labs, _ = zip(*batch)
    lengths = [e.size(0) for e in embs]
    embs_padded = pad_sequence(embs, batch_first=True)
    labels = torch.tensor(labs, dtype=torch.long)
    return embs_padded, labels, lengths


def collate_fn_val(batch):
    embs, labs, clips = zip(*batch)
    lengths = [e.size(0) for e in embs]
    embs_padded = pad_sequence(embs, batch_first=True)
    labels = torch.tensor(labs, dtype=torch.long)
    return embs_padded, labels, lengths, clips


class CREMADDataset(Dataset):
    def __init__(self, clip_names, labels, ssl, cache_dir, device, target_sr=16000):
        self.clips     = clip_names
        self.labels    = labels
        self.ssl       = ssl
        self.cache_dir = cache_dir
        self.device    = device
        self.target_sr = target_sr
        self.audio_dir = DEFAULT_AUDIO_DIR

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip = self.clips[idx]
        cache_path = os.path.join(self.cache_dir, clip + '_layer10.npy')
        emb = np.load(cache_path)
        emb = torch.from_numpy(emb).float()
        label = self.labels[idx]
        return emb, label, clip

def precompute_feats(args):
    # load & filter out 'D'
    df = pd.read_csv(args.label_file).drop_duplicates(subset="clipName")
    clips = df["clipName"].values

    device = args.device
    ssl = SSLModel(device)
    ssl.model.eval().to(device)

    class RawDataset(Dataset):
        def __init__(self, clips, audio_dir, target_sr):
            self.clips = clips
            self.audio_dir = audio_dir
            self.target_sr = target_sr

        def __len__(self):
            return len(self.clips)

        def __getitem__(self, idx):
            clip = self.clips[idx]
            path = os.path.join(self.audio_dir, clip + ".wav")
            wav, sr = torchaudio.load(path)
            wav = wav.mean(dim=0)
            if sr != self.target_sr:
                wav = torchaudio.functional.resample(wav, sr, self.target_sr)
            return clip, wav

    raw_ds = RawDataset(clips, args.audio_dir, args.target_sr)
    loader = DataLoader(raw_ds, batch_size=1, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    with torch.no_grad():
        for clip, wav in loader:
            clip = clip[0]
            wav = wav[0].to(device)
            out = ssl.model(wav.unsqueeze(0), mask=False, features_only=True)
            final_feat = out["x"]
            layer_results = out["layer_results"]
            for i, hid in enumerate(layer_results):
                tensor = hid[0] if isinstance(hid, tuple) else hid
                arr = tensor.squeeze(1).cpu().numpy()
                np.save(os.path.join(args.cache_dir, f"{clip}_layer{i}.npy"), arr)
            final_arr = final_feat.squeeze(0).cpu().numpy()
            np.save(os.path.join(args.cache_dir, f"{clip}_final.npy"), final_arr)

    print("✅ Cached all layer outputs in:", args.cache_dir)

def vicreg_loss(z1: torch.Tensor):
    """
    VICReg loss: variance, covariance terms.
    """
    eps = 1e-6
    std_target = 1.0
    def variance_term(z):
        std = torch.sqrt(z.var(dim=0) + eps)
        return torch.mean(F.relu(std_target - std))

    # covariance loss
    def covariance_term(z):
        B, D = z.size()
        z = z - z.mean(dim=0)
        cov = (z.T @ z) / (B - 1)  # (D, D)
        off_diag_mask = ~torch.eye(D, device=z.device, dtype=torch.bool)
        return (cov[off_diag_mask] ** 2).sum() / D
    return variance_term(z1), covariance_term(z1)



def train_loop(args):
    # load & filter out 'D'
    df = pd.read_csv(args.label_file).drop_duplicates(subset="clipName")
    clips = df['clipName'].values
    labels = df['dispEmo'].values
    le = LabelEncoder().fit(labels)
    y_all = le.transform(labels)

    # random train/val split
    train_clips, val_clips, train_y, val_y = train_test_split(
        clips, y_all,
        test_size=args.val_split,
        random_state=args.seed,
        stratify=y_all
    )

    device = args.device
    ssl = SSLModel(device)
    ssl.model.eval().to(device)

    train_ds = CREMADDataset(train_clips, train_y, ssl, args.cache_dir, device, target_sr=args.target_sr)
    val_ds   = CREMADDataset(val_clips,   val_y,   ssl, args.cache_dir, device, target_sr=args.target_sr)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn_train, num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate_fn_val,   num_workers=args.num_workers, pin_memory=True)

    model = EmotionClassifier(ssl, feat_dim=ssl.out_dim, num_classes=len(le.classes_))
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.01)
    gamma = (args.end_lr / args.lr) ** (1.0 / (args.epochs - 1))
    scheduler = ExponentialLR(optimizer, gamma=gamma)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_path = os.path.join('checkpoints', 'best_model.pth')
    os.makedirs('checkpoints', exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        # ---- train ----
        model.train()
        running_loss = 0.0
        all_labels, all_preds = [], []

        for embs, labels_batch, lengths in train_loader:
            embs, labels_batch = embs.to(device), labels_batch.to(device)
            optimizer.zero_grad()
            logits, cls_repr = model(embs, lengths)
            ce_loss = criterion(logits, labels_batch)
            var_loss, co_loss = vicreg_loss(cls_repr)
            loss = ce_loss + co_loss + var_loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * embs.size(0)
            preds = logits.argmax(dim=1)
            all_labels.extend(labels_batch.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

        train_acc = np.mean(np.array(all_preds) == np.array(all_labels))
        avg_loss = running_loss / len(train_ds)
        print(f"Epoch {epoch} — loss: {avg_loss:.4f} — train acc: {train_acc:.4f}")
        print(classification_report(all_labels, all_preds, target_names=le.classes_))

        # ---- validation ----
        model.eval()
        val_labels, val_preds = [], []
        print("Validation misclassifications:")
        with torch.no_grad():
            for embs, labels_batch, lengths, clip_names in val_loader:
                embs, labels_batch = embs.to(device), labels_batch.to(device)
                logits, _ = model(embs, lengths)
                preds = logits.argmax(dim=1)

                for clip, true_lbl, pred_lbl in zip(clip_names, labels_batch.cpu().tolist(), preds.cpu().tolist()):
                    if true_lbl != pred_lbl:
                        audio_path = os.path.join(args.audio_dir, clip + '.wav')
                        print(f"Misclassified: {audio_path} | True: {le.classes_[true_lbl]} | Pred: {le.classes_[pred_lbl]}")

                val_labels.extend(labels_batch.cpu().tolist())
                val_preds.extend(preds.cpu().tolist())

        val_acc = np.mean(np.array(val_preds) == np.array(val_labels))
        print(f" — val acc: {val_acc:.4f}")
        print(classification_report(val_labels, val_preds, target_names=le.classes_))
        print(confusion_matrix(val_labels, val_preds))
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)
            print(f"🔖 New best model saved to {best_path}")
        
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"→ End of epoch {epoch}, lr is now {current_lr:.8e}\n")

    print(f"\n🏁 Training complete. Best val acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",       choices=["precompute", "train"], required=True)
    parser.add_argument("--audio_dir",  default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--label_file", default=DEFAULT_LABEL_FILE)
    parser.add_argument("--cache_dir",  default='cached_feats')
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--num_workers",type=int,   default=4)
    parser.add_argument('--lr', type=float, default=2.5e-4)
    parser.add_argument('--end_lr', type=float, default=1e-6)
    parser.add_argument("--target_sr",  type=int,   default=16000)
    parser.add_argument("--val_split",  type=float, default=0.1, help="fraction of data to use for validation")
    parser.add_argument("--seed",       type=int,   default=42,  help="random seed for split")
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == "precompute":
        precompute_feats(args)
    else:
        train_loop(args)

