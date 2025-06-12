#!/usr/bin/env python3

import os
import torch
import numpy as np
import pandas as pd
import h5py
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, cohen_kappa_score, matthews_corrcoef
)

from src.models.mil_models_multiclass import Attention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Using device: {device}")

CLASS_NAMES = ["WD", "MD", "PD"]
NUM_CLASSES = 3


# ---------------- Dataset Class ---------------- #
class MulticlassOSCCDataset:
    def __init__(self, hdf5_file, test_slide_ids):
        self.bags = []
        self.coords = []
        self.labels = []
        self.case_ids = []
        self.slide_ids = []
        self.load_bags(hdf5_file, test_slide_ids)

    def load_bags(self, hdf5_file, test_slide_ids):
        with h5py.File(hdf5_file, 'r') as h5f:
            for sid in test_slide_ids:
                if sid in h5f:
                    group = h5f[sid]
                    self.bags.append(torch.tensor(group["embeddings"][:]))
                    self.coords.append(torch.tensor(group["coords"][:]))
                    self.labels.append(int(group["label"][()]))
                    self.case_ids.append(group.attrs["slide_name"])
                    self.slide_ids.append(sid)
        print(f"✅ Loaded {len(self.bags)} samples from HDF5")


# ---------------- Prediction + Attention Saving ---------------- #
def save_attention_scores(bag_coords, attention_scores, path, output_dir):
    df = pd.DataFrame({
        "row": bag_coords[:, 0],
        "col": bag_coords[:, 1],
        "attention_score": attention_scores
    })
    df.to_csv(os.path.join(output_dir, f"{path}_attention.csv"), index=False)


def predict_on_dataset(dataset, model, output_dir):
    true_labels, pred_labels = [], []

    for bag, coords, label, sid in zip(dataset.bags, dataset.coords, dataset.labels, dataset.slide_ids):
        bag = bag.unsqueeze(0).to(device)  # Add batch dim

        with torch.no_grad():
            _, pred, attn = model(bag)
        pred_label = int(pred.item())

        # Save attention scores
        save_attention_scores(coords.cpu().numpy(), attn.squeeze().cpu().numpy(), sid, output_dir)

        true_labels.append(label)
        pred_labels.append(pred_label)

    return true_labels, pred_labels


# ---------------- Metric Computation ---------------- #
def compute_and_save_metrics(true, pred, out_dir):
    acc = accuracy_score(true, pred)
    precision = precision_score(true, pred, average='macro')
    recall = recall_score(true, pred, average='macro')
    f1 = f1_score(true, pred, average='macro')
    kappa = cohen_kappa_score(true, pred)
    mcc = matthews_corrcoef(true, pred)
    report = classification_report(true, pred, target_names=CLASS_NAMES, digits=4, output_dict=True)

    # Save metrics
    pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "Cohen's Kappa", "MCC"],
        "Value": [acc, precision, recall, f1, kappa, mcc]
    }).to_csv(os.path.join(out_dir, "metrics_summary.csv"), index=False)

    # Save classification report
    pd.DataFrame(report).transpose().to_csv(os.path.join(out_dir, "classification_report.csv"))

    # Confusion matrix
    cm = confusion_matrix(true, pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"))
    plt.close()
    print(f"📈 Saved metrics and confusion matrix to {out_dir}")


# ---------------- Per-Fold Evaluation ---------------- #
def process_all_folds(run_dir, output_base_dir, hdf5_file, test_csv):
    test_df = pd.read_csv(test_csv)
    test_ids = test_df["Slide_ID"].tolist()

    for fold in range(1, 4):
        print(f"\n🔍 Evaluating Fold {fold}")
        model_path = os.path.join(run_dir, f"fold_{fold}/models/model_epoch_50.pth")
        output_dir = os.path.join(output_base_dir, f"fold_{fold}")
        os.makedirs(output_dir, exist_ok=True)

        model = Attention(L=512, D=128, K=1, num_classes=NUM_CLASSES).to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        dataset = MulticlassOSCCDataset(hdf5_file, test_ids)
        true, pred = predict_on_dataset(dataset, model, output_dir)
        compute_and_save_metrics(true, pred, output_dir)


# ---------------- Run Main ---------------- #
if __name__ == "__main__":
    HDF5_FILE = "/data/features/features_from_resnet50_oscc_tcia/oscc_features_resnet50.hdf5"
    TEST_CSV = "data/features/features_from_resnet50_oscc_tcia/oscc-class.csv"
    RUN_DIR = "/data/outputs/multi_class/results"
    OUTPUT_DIR = os.path.join(RUN_DIR, "test_results_on_oscc")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    process_all_folds(RUN_DIR, OUTPUT_DIR, HDF5_FILE, TEST_CSV)
