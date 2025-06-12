import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, cohen_kappa_score, matthews_corrcoef
)
import seaborn as sns
import matplotlib.pyplot as plt
import h5py
# from mil_models import ModBagClassifier, Attention
from src.models.mil_models_binary import ModBagClassifier, Attention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Define the dataset class
class GeneExpressionDataset:
    def __init__(self, hdf5_file, label):
        self.label = label
        self.bags, self.coords, self.labels, self.slide_ids, self.case_ids = self.get_bag_embeddings(hdf5_file, label)

    def get_bag_embeddings(self, hdf5_file, label):
        bags = []
        coords = []
        labels = []
        slide_ids = []
        case_ids = []

        with h5py.File(hdf5_file, 'r') as file:
            for group in file.values():
                bags.append(torch.tensor(np.array(group["embeddings"])))
                coords.append(torch.tensor(np.array(group["coords"])))
                labels.append(torch.tensor(int(group["label"][()])))
                slide_ids.append(group.attrs["slide_id"])
                case_ids.append(group.attrs["path"])
        
        print(f"Loaded {len(bags)} bags from {hdf5_file}")
        return bags, coords, labels, slide_ids, case_ids

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, index):
        return self.bags[index], self.coords[index], self.labels[index], self.slide_ids[index], self.case_ids[index]


# Function to save attention scores
def save_attention_scores(bag_coords, attention_scores, path, output_dir):
    data = {
        "row": bag_coords[:, 0].tolist(),
        "col": bag_coords[:, 1].tolist(),
        "attention_score": attention_scores.tolist()
    }
    csv_path = os.path.join(output_dir, f"{path}.csv")
    pd.DataFrame(data).to_csv(csv_path, index=False)
    print(f"Saved attention scores to {csv_path}")


# Function to process bags and compute metrics
def process_bags(hdf5_file, test_csv, model, output_dir):
    """
    Process bags, filter by slide IDs, compute predictions using Y_hat, and save results.
    """
    # Load test slide IDs
    test_slides = pd.read_csv(test_csv)["Slide_ID"].tolist()

    # Initialize the dataset
    dataset = GeneExpressionDataset(hdf5_file, label="label")
    # print("")

    # Extract bags, coordinates, labels, and paths
    bags = dataset.bags
    coords_list = dataset.coords
    labels = dataset.labels
    case_ids = dataset.case_ids

    print("case_id", case_ids)
    
    # Initialize lists for true and predicted labels
    true_labels = []
    predicted_labels = []
    prediction_probs = []

    for bag, coords, label, case_id in zip(bags, coords_list, labels, case_ids):
        if case_id in test_slides:
            print("label", label.item())
            
            # Move bag to device
            bag = bag.to(device)

            # Get attention scores and predictions
            with torch.no_grad():
                _, Y_hat, attention_scores = model(bag)
            attention_scores = attention_scores.cpu().numpy().flatten()

            # Save attention scores and coordinates for this bag
            save_attention_scores(coords.cpu().numpy(), attention_scores, case_id, output_dir)

            # Append true and predicted labels
            true_labels.append(label.item())  # Ensure it's a scalar
            predicted_labels.append(int(Y_hat.item()))  # Ensure it's an integer
            prediction_probs.append(float(Y_hat.cpu().numpy()))  # Save probability

    # Save predictions to CSV
    predictions_df = pd.DataFrame({
        "True Label": true_labels,
        "Predicted Label": predicted_labels,
        "Prediction Probability": prediction_probs
    })
    predictions_csv_path = os.path.join(output_dir, "predictions.csv")
    predictions_df.to_csv(predictions_csv_path, index=False)
    print(f"Saved predictions to {predictions_csv_path}")

    return true_labels, predicted_labels

def compute_and_save_metrics(true_labels, predicted_labels, output_dir):
    """
    Compute metrics, generate classification report, and save confusion matrix.
    """
    # Compute metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    mcc = matthews_corrcoef(true_labels, predicted_labels)
    kappa = cohen_kappa_score(true_labels, predicted_labels)
    report = classification_report(true_labels, predicted_labels, digits=4, output_dict=True)

    # Save metrics as a separate CSV
    metrics_data = {
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "MCC", "Cohen's Kappa"],
        "Value": [accuracy, precision, recall, f1, mcc, kappa]
    }
    metrics_df = pd.DataFrame(metrics_data)
    metrics_csv_path = os.path.join(output_dir, "metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"Saved metrics to {metrics_csv_path}")

    # Save classification report as a separate CSV
    report_df = pd.DataFrame(report).transpose()
    report_csv_path = os.path.join(output_dir, "classification_report.csv")
    report_df.to_csv(report_csv_path)
    print(f"Saved classification report as CSV to {report_csv_path}")

    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)

    # Save confusion matrix as text
    cm_text_path = os.path.join(output_dir, "confusion_matrix.txt")
    with open(cm_text_path, "w") as f:
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(cm, separator=', '))
    print(f"Saved confusion matrix as text to {cm_text_path}")

    # Plot and save confusion matrix as an image
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=["Normal", "OSCC"], yticklabels=["Normal", "OSCC"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    cm_image_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_image_path)
    plt.show()
    print(f"Saved confusion matrix as image to {cm_image_path}")


# Function to iterate through folds
def process_all_folds(input_dir, output_base_dir):
    """
    Iterate over all folds, load respective models, process their data, and save results.
    """
    for fold_idx in range(1, 4): 
        
        fold_input_dir = os.path.join(input_dir, f"fold_{fold_idx}")
        hdf5_file = os.path.join("/data/features/features_from_resnet50/Binary_features_resnet50.hdf5")
        test_csv = os.path.join("/data/features/features_from_resnet50/fold_csv/External_test_IDs_binary.csv")
        model_path = os.path.join(fold_input_dir, "models/model_epoch_50.pth")

        # Create output directory for this fold
        fold_output_dir = os.path.join(output_base_dir, f"fold_{fold_idx}")
        os.makedirs(fold_output_dir, exist_ok=True)

        # Load the model for this fold
        print(f"Loading model for Fold {fold_idx} from {model_path}")
        model = Attention().to(device)
        # model = Attention().to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        print(f"Processing Fold {fold_idx}...")
        true_labels, predicted_labels = process_bags(hdf5_file, test_csv, model, fold_output_dir)
        compute_and_save_metrics(true_labels, predicted_labels, fold_output_dir)


# Main paths
main_dir = "/data/outputs/binary/results/"  
output_base_dir = os.path.join(main_dir, "test-results-binary")  
os.makedirs(output_base_dir, exist_ok=True)

# Process all folds
process_all_folds(main_dir, output_base_dir)