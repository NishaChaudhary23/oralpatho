#!/usr/bin/env python3

import os
import torch
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, cohen_kappa_score, matthews_corrcoef
)
from src.models.mil_models_multiclass import Attention

# ---------------- SETUP ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Using device: {device}")

HDF5_FILE = "/data/features_organized_by_class/oscc_grading_features_resnet50.hdf5"
VAL_CSV = "/data/features/fold_csv_multiclass/Val_IDs_fold0.csv"
MODEL_PATH = "/data/outputs/multi_class/results/fold_1/models/best_model.pth"
OUTPUT_DIR = "/data/outputs/multi_class/results/fold_1/test_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
ATTN_DIR = os.path.join(OUTPUT_DIR, "attention_csv")
os.makedirs(ATTN_DIR, exist_ok=True)

CLASS_NAMES = ["WD", "MD", "PD"]
label_remap = {2: 0, 3: 1, 4: 2}

# ---------------- Dataset ----------------
class SlideDataset:
    def __init__(self, h5_path, slide_ids):
        self.h5_path = h5_path
        self.slide_ids = slide_ids

    def __len__(self):
        return len(self.slide_ids)

    def __getitem__(self, idx):
        sid = self.slide_ids[idx]
        with h5py.File(self.h5_path, 'r') as h5f:
            group = h5f[sid]
            x = torch.tensor(group["embeddings"][:], dtype=torch.float32)
            label = int(group["label"][()])
            coords = group["coords"][:]
        return x, label, coords, sid

# ---------------- Save Attention ----------------
def save_attention_map(slide_id, coords, attn_scores):
    df = pd.DataFrame({
        "row": coords[:, 0],
        "col": coords[:, 1],
        "attention": attn_scores
    })
    df.to_csv(os.path.join(ATTN_DIR, f"{slide_id}.csv"), index=False)

# ---------------- Run Inference ----------------
val_ids = pd.read_csv(VAL_CSV)["Slide_ID"].tolist()
dataset = SlideDataset(HDF5_FILE, val_ids)

model = Attention(L=512, D=128, K=1, num_classes=3).to(device)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

true_labels = []
pred_labels = []
probabilities = []

print(f"Inference on {len(dataset)} slides")

for x, label, coords, sid in dataset:
    x = x.unsqueeze(0).to(device)  # Shape: (1, N, L)
    with torch.no_grad():
        prob, y_hat, A = model(x)

    true_labels.append(label)
    pred_labels.append(int(y_hat.item()))
    probabilities.append(prob.cpu().numpy().flatten().tolist())
    save_attention_map(sid, coords, A.cpu().numpy().flatten())

# ---------------- Remap Labels ----------------
mapped_true = [label_remap.get(lbl, -1) for lbl in true_labels]
valid_indices = [i for i, val in enumerate(mapped_true) if val != -1]
mapped_true = [mapped_true[i] for i in valid_indices]
mapped_pred = [pred_labels[i] for i in valid_indices]
mapped_probs = [probabilities[i] for i in valid_indices]
mapped_ids = [val_ids[i] for i in valid_indices]

# ---------------- Save Predictions ----------------
df_pred = pd.DataFrame({
    "Slide_ID": mapped_ids,
    "True_Label": mapped_true,
    "Predicted_Label": mapped_pred,
    "Probabilities": mapped_probs
})
df_pred.to_csv(os.path.join(OUTPUT_DIR, "predictions.csv"), index=False)
print(" Saved predictions.csv")

# ---------------- Save Metrics ----------------
acc = accuracy_score(mapped_true, mapped_pred)
prec = precision_score(mapped_true, mapped_pred, average='macro', zero_division=0)
rec = recall_score(mapped_true, mapped_pred, average='macro', zero_division=0)
f1 = f1_score(mapped_true, mapped_pred, average='macro', zero_division=0)
kappa = cohen_kappa_score(mapped_true, mapped_pred)
mcc = matthews_corrcoef(mapped_true, mapped_pred)
report = classification_report(mapped_true, mapped_pred, target_names=CLASS_NAMES, digits=4)
conf = confusion_matrix(mapped_true, mapped_pred)

# Save classification report
with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
    f.write(report)

# Save confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(conf, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
plt.close()

# Save metrics CSV
pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "Cohen's Kappa", "MCC"],
    "Value": [acc, prec, rec, f1, kappa, mcc]
}).to_csv(os.path.join(OUTPUT_DIR, "metrics.csv"), index=False)

print("Metrics, classification report & confusion matrix saved.")
