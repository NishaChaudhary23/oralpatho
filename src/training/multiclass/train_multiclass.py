"""
Train Multiclass MIL Model (e.g., WD vs MD vs PD OSCC)
"""


#  
import os
import time
import csv
import torch
import pandas as pd
import numpy as np
import h5py
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, cohen_kappa_score, matthews_corrcoef
)
from sklearn.utils.class_weight import compute_class_weight

#from mil_models import AttentionTopK
#from wsi_dataset import TumorEmbeddingDataset
#from collate_fn import tumor_pad_collate_fn
from src.models.mil_models_multiclass import AttentionTopK
from src.datasets.wsi_datasets_multiclass import TumorEmbeddingDataset
from src.utils.collate_fn_multiclass import tumor_pad_collate_fn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Using device: {device}")

HDF5_FILE = "/path/to/features/expanded_sections_allpatches.hdf5"
fold_csv_dir = "/path/to/features/fold_csv_multiclass/new-split-PD"
output_dir = "/path/to/outputs/multi_class/results"
os.makedirs(output_dir, exist_ok=True)

label_map = {2: 0, 3: 1, 4: 2}
class_names = ["WD", "MD", "PD"]
num_epochs = 50
lr = 5e-5
weight_decay = 1e-4
top_k = 70
gamma = 2.5

class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        prob = torch.exp(log_prob)
        focal = (1 - prob) ** self.gamma
        loss = F.nll_loss(focal * log_prob, target, weight=self.weight)
        return loss

for fold in range(3):
    print(f"\n========== Fold {fold + 1}/3 ==========")
    train_df = pd.read_csv(os.path.join(fold_csv_dir, f"Train_IDs_fold{fold}.csv"))
    val_df = pd.read_csv(os.path.join(fold_csv_dir, f"Val_IDs_fold{fold}.csv"))

    train_ids = train_df["Slide_ID"].tolist()
    val_ids = val_df["Slide_ID"].tolist()

    with h5py.File(HDF5_FILE, 'r') as h5f:
        available_ids = list(h5f.keys())
    train_ids = [i for i in train_ids if i in available_ids]
    val_ids = [i for i in val_ids if i in available_ids]

    print(f"Fold {fold} — Train: {len(train_ids)} | Val: {len(val_ids)}")

    train_dataset = TumorEmbeddingDataset(train_ids, HDF5_FILE, label_map)
    val_dataset = TumorEmbeddingDataset(val_ids, HDF5_FILE, label_map)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=tumor_pad_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=tumor_pad_collate_fn)

    y_fold = train_df["label"].map(label_map).dropna().astype(int).values
    #class_weights = compute_class_weight('balanced', classes=np.unique(y_fold), y=y_fold)
    class_weights = [1.0, 1.0, 5.0]  # Manually boost PD class weight
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    fold_dir = os.path.join(output_dir, f"fold_{fold + 1}")
    os.makedirs(fold_dir, exist_ok=True)
    os.makedirs(os.path.join(fold_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(fold_dir, "confusion_matrices"), exist_ok=True)
    os.makedirs(os.path.join(fold_dir, "classification_reports"), exist_ok=True)

    metrics_csv = os.path.join(fold_dir, "epoch_metrics.csv")
    with open(metrics_csv, "w", newline="") as f:
        csv.writer(f).writerow([
            'epoch', 'train_loss', 'val_loss',
            'train_acc', 'train_prec', 'train_recall', 'train_f1', 'train_kappa', 'train_mcc',
            'val_acc', 'val_prec', 'val_recall', 'val_f1', 'val_kappa', 'val_mcc'
        ])

    model = AttentionTopK(L=512, D=128, K=1, num_classes=3, top_k=top_k).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = FocalLoss(gamma=gamma, weight=weights_tensor)

    best_val_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_preds, train_labels = 0.0, [], []
        for x, y, mask in train_loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            loss, _, _ = model.calculate_objective(x, y, mask=mask, loss_fn=loss_fn)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, y_hat, _ = model(x, mask=mask)
            train_preds.append(y_hat.item())
            train_labels.append(y.item())

        model.eval()
        val_loss, val_preds, val_labels = 0.0, [], []
        with torch.no_grad():
            for x, y, mask in val_loader:
                x, y, mask = x.to(device), y.to(device), mask.to(device)
                loss, _, _ = model.calculate_objective(x, y, mask=mask, loss_fn=loss_fn)
                val_loss += loss.item()
                _, y_hat, _ = model(x, mask=mask)
                val_preds.append(y_hat.item())
                val_labels.append(y.item())

        def get_metrics(true, pred, avg='macro'):
            return (
                accuracy_score(true, pred),
                precision_score(true, pred, average=avg, zero_division=0),
                recall_score(true, pred, average=avg, zero_division=0),
                f1_score(true, pred, average=avg, zero_division=0),
                cohen_kappa_score(true, pred),
                matthews_corrcoef(true, pred)
            )

        train_metrics = get_metrics(train_labels, train_preds)
        val_metrics = get_metrics(val_labels, val_preds)

        print(f"Epoch {epoch + 1}/{num_epochs} | Train Acc: {train_metrics[0]:.4f} | Val Acc: {val_metrics[0]:.4f}")

        with open(metrics_csv, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch + 1,
                train_loss, val_loss,
                *train_metrics,
                *val_metrics
            ])

        if val_metrics[0] > best_val_acc:
            best_val_acc = val_metrics[0]
            torch.save(model.state_dict(), os.path.join(fold_dir, "models", "best_model.pth"))
            print(" Saved best model")

            # Save confusion matrix & classification report for best epoch
            cm = confusion_matrix(val_labels, val_preds)
            cr = classification_report(val_labels, val_preds, target_names=class_names, digits=4)

            with open(os.path.join(fold_dir, "confusion_matrices", f"best_cm.txt"), "w") as f:
                f.write(str(cm))

            with open(os.path.join(fold_dir, "classification_reports", f"best_cr.txt"), "w") as f:
                f.write(cr)

print("\n All folds completed and saved.")