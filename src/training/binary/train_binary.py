"""
Train Binary MIL Model (Normal vs OSCC)
"""

#=======================================================================================================
#
# export PYTHONPATH=./src:$PYTHONPATH
# python src/training/binary/train_binary.py ...


import os
import csv
import time
import pandas as pd
import numpy as np
import torch
from collections import Counter
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, matthews_corrcoef, cohen_kappa_score
)
from src.datasets.wsi_datasets import TumorEmbeddingDataset, tumor_pad_collate_fn
from src.models.mil_models import Attention
#from wsi_datasets import TumorEmbeddingDataset, tumor_pad_collate_fn
#from mil_models import ModBagClassifier, Attention
from sklearn.utils.class_weight import compute_class_weight


# ======================= Device Setup ======================= #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ======================= Dataset and DataLoader ======================= #
dataset_path = "/path/to/features/Binary_features_resnet50.hdf5"
tumor_dataset = TumorEmbeddingDataset(dataset_path)

# ======================= Directory Paths ======================= #
fold_ids_dir = "/path/to/features/features_from_resnet50/fold_csv/"
output_dir = "/path/to/outputs/binary/results"
os.makedirs(output_dir, exist_ok=True)

num_epochs = 50
LR = 1e-4
REG = 0.000152652510413

# ======================== Initialize_weights ===============================#
def initialize_weights(module):
    """Custom weight initialization for layers."""
    if isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')  # Kaiming for ReLU
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')  # He initialization
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.ones_(module.weight)  # BatchNorm weights initialized to 1
        nn.init.zeros_(module.bias)  # BatchNorm biases initialized to 0
    elif isinstance(module, nn.Embedding):
        nn.init.uniform_(module.weight, -0.1, 0.1)  # Uniform initialization for embeddings

# ======================= Training and Validation ======================= #
for fold in range(3):
    print(f"\n========== Fold {fold + 1}/3 ==========")

    # Load train and validation IDs for the current fold
    train_ids_csv = os.path.join(fold_ids_dir, f"Train_IDs_fold{fold}.csv")
    val_ids_csv = os.path.join(fold_ids_dir, f"Val_IDs_fold{fold}.csv")

    train_ids_df = pd.read_csv(train_ids_csv)
    val_ids_df = pd.read_csv(val_ids_csv)

    train_slide_ids = train_ids_df['Slide_ID'].tolist()
    val_slide_ids = val_ids_df['Slide_ID'].tolist()


    labels_ = train_ids_df['label'].tolist()  
    class_weights = compute_class_weight(class_weight='balanced',   classes=np.unique(labels_), y=labels_  )

    # Convert to a PyTorch tensor
    # weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    pos_weight = class_weights[1]  # Extract weight for class 
    print("class_weights",class_weights)
    
    pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32).to(device)
    print("Class weights:", pos_weight_tensor)
    
    # Filter dataset to include only slides in the respective IDs
    train_indices = [i for i, slide_id in enumerate(tumor_dataset.paths) if slide_id in train_slide_ids]
    val_indices = [i for i, slide_id in enumerate(tumor_dataset.paths) if slide_id in val_slide_ids]

    train_dataset = Subset(tumor_dataset, train_indices)
    val_dataset = Subset(tumor_dataset, val_indices)

    print(f"Train samples: {len(train_indices)}, Validation samples: {len(val_indices)}")

    # Create fold-specific directory structure
    fold_dir = os.path.join(output_dir, f"fold_{fold + 1}")
    os.makedirs(fold_dir, exist_ok=True)
    confusion_dir = os.path.join(fold_dir, "confusion_matrices")
    classification_dir = os.path.join(fold_dir, "classification_reports")
    model_dir = os.path.join(fold_dir, "models")
    os.makedirs(confusion_dir, exist_ok=True)
    os.makedirs(classification_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Create fold-specific CSV file for metrics
    metrics_csv_path = os.path.join(fold_dir, "epoch_metrics.csv")
    with open(metrics_csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            'epoch', 'time_seconds', 'train_samples', 'val_samples', 'train_loss',
            'train_accuracy', 'train_precision', 'train_recall', 'train_f1',
            'train_kappa', 'train_mcc', 'val_loss', 'val_accuracy', 'val_precision',
            'val_recall', 'val_f1', 'val_kappa', 'val_mcc'
        ])

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=1, collate_fn=tumor_pad_collate_fn, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=tumor_pad_collate_fn, shuffle=False)

    # Model, Criterion, Optimizer, Scheduler
    model = Attention().to(device)
    model.apply(initialize_weights)  # Apply custom weight initialization
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)

    # loss_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min', factor=0.5, patience=3,  verbose=True)
    # acc_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max', factor=0.5, patience=3,  verbose=True)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    # Set the accumulation steps
    accumulation_steps = 4

    for epoch in range(num_epochs):
        start_time = time.time()
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Initialize metrics
        epoch_metrics = {
            "train": {"loss": 0, "accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "kappa": 0, "mcc": 0},
            "val": {"loss": 0, "accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "kappa": 0, "mcc": 0},
        }

        # ---------- Training ---------- #
        model.train()
        train_loss = 0.0
        all_train_predictions, all_train_labels = [], []
        optimizer.zero_grad()  

        for batch_idx, batch in enumerate(train_loader):
            padded_bags, padded_coords, padded_labels, separate_paths, is_aug, mask = batch
            padded_bags = padded_bags.to(device)
            padded_labels = padded_labels.float().to(device) 
            mask = mask.to(device)

            # Forward pass
            Y_prob, Y_hat, _ = model(padded_bags, mask)  
            slide_label = padded_labels[0].item()
            loss = criterion(Y_prob, padded_labels[0].unsqueeze(0).unsqueeze(1).float()) / accumulation_steps
            loss.backward() 
            train_loss += loss.item() * accumulation_steps  
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()  

            all_train_predictions.append(Y_hat.item())
            all_train_labels.append(slide_label)

        epoch_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch + 1} Train Loss: {epoch_train_loss:.4f}")
        
        # Calculate training metrics
        epoch_metrics["train"]["loss"] = epoch_train_loss
        epoch_metrics["train"]["accuracy"] = accuracy_score(all_train_labels, all_train_predictions)
        epoch_metrics["train"]["precision"] = precision_score(all_train_labels, all_train_predictions, average='binary')
        epoch_metrics["train"]["recall"] = recall_score(all_train_labels, all_train_predictions, average='binary')
        epoch_metrics["train"]["f1"] = f1_score(all_train_labels, all_train_predictions, average='binary')
        epoch_metrics["train"]["kappa"] = cohen_kappa_score(all_train_labels, all_train_predictions)
        epoch_metrics["train"]["mcc"] = matthews_corrcoef(all_train_labels, all_train_predictions)

        # ---------- Validation ---------- #
        model.eval()
        val_loss = 0.0
        all_val_predictions, all_val_labels = [], []

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                padded_bags, padded_coords, padded_labels, separate_paths, is_aug, mask = batch
                padded_bags = padded_bags.to(device)
                padded_labels = padded_labels.float().to(device)  
                mask = mask.to(device)

                Y_prob, Y_hat, _ = model(padded_bags, mask)
                slide_label = padded_labels[0].item()
                loss = criterion(Y_prob, padded_labels[0].unsqueeze(0).unsqueeze(1).float()) 
                val_loss += loss.item()
                all_val_predictions.append(Y_hat.item())
                all_val_labels.append(slide_label)

        # Calculate validation metrics
        epoch_metrics["val"]["loss"] = val_loss / len(val_loader)
        epoch_metrics["val"]["accuracy"] = accuracy_score(all_val_labels, all_val_predictions)
        epoch_metrics["val"]["precision"] = precision_score(all_val_labels, all_val_predictions, average='binary')
        epoch_metrics["val"]["recall"] = recall_score(all_val_labels, all_val_predictions, average='binary')
        epoch_metrics["val"]["f1"] = f1_score(all_val_labels, all_val_predictions, average='binary')
        epoch_metrics["val"]["kappa"] = cohen_kappa_score(all_val_labels, all_val_predictions)
        epoch_metrics["val"]["mcc"] = matthews_corrcoef(all_val_labels, all_val_predictions)

        # Print epoch metrics
        print(f"Train Accuracy: {epoch_metrics['train']['accuracy']:.4f}, Validation Accuracy: {epoch_metrics['val']['accuracy']:.4f}")

        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_accuracy = accuracy_score(all_val_labels, all_val_predictions)
        print(f"Epoch {epoch + 1} Validation Loss: {epoch_val_loss:.4f}")

        # Scheduler Step
        scheduler.step(epoch_metrics["val"]["loss"])
        # loss_scheduler.step(epoch_val_loss)
        # acc_scheduler.step(epoch_val_accuracy)

        print(f"Learning Rate: {optimizer.param_groups[0]['lr']}")

        end_time = time.time()
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch + 1} completed in {end_time - start_time:.2f}s")

       # ---------- Save Metrics to CSV ---------- #
        with open(metrics_csv_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1, round(epoch_time, 2), len(train_indices), len(val_indices),
                epoch_metrics["train"]["loss"], epoch_metrics["train"]["accuracy"],
                epoch_metrics["train"]["precision"], epoch_metrics["train"]["recall"],
                epoch_metrics["train"]["f1"], epoch_metrics["train"]["kappa"],
                epoch_metrics["train"]["mcc"], epoch_metrics["val"]["loss"],
                epoch_metrics["val"]["accuracy"], epoch_metrics["val"]["precision"],
                epoch_metrics["val"]["recall"], epoch_metrics["val"]["f1"],
                epoch_metrics["val"]["kappa"], epoch_metrics["val"]["mcc"]
            ])

        # ---------- Save Confusion Matrix and Classification Report ---------- #
        cm_train = confusion_matrix(all_train_labels, all_train_predictions)
        cm_val = confusion_matrix(all_val_labels, all_val_predictions)
        cr_train = classification_report(all_train_labels, all_train_predictions, digits=4)
        cr_val = classification_report(all_val_labels, all_val_predictions, digits=4)

        cm_train_path = os.path.join(confusion_dir, f"train_epoch_{epoch + 1}.txt")
        cm_val_path = os.path.join(confusion_dir, f"val_epoch_{epoch + 1}.txt")
        cr_train_path = os.path.join(classification_dir, f"train_epoch_{epoch + 1}.txt")
        cr_val_path = os.path.join(classification_dir, f"val_epoch_{epoch + 1}.txt")

        with open(cm_train_path, "w") as f:
            f.write(f"Confusion Matrix (Train) - Epoch {epoch + 1}\n")
            f.write(str(cm_train))
        with open(cm_val_path, "w") as f:
            f.write(f"Confusion Matrix (Validation) - Epoch {epoch + 1}\n")
            f.write(str(cm_val))
        with open(cr_train_path, "w") as f:
            f.write(f"Classification Report (Train) - Epoch {epoch + 1}\n")
            f.write(cr_train)
        with open(cr_val_path, "w") as f:
            f.write(f"Classification Report (Validation) - Epoch {epoch + 1}\n")
            f.write(cr_val)

        # ---------- Save Model ---------- #
        
        model_path = os.path.join(model_dir, f"model_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
