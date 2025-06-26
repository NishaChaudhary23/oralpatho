import torch
import h5py
import numpy as np
import os
from mil_models import ModBagClassifier, Attention


# ------------------- Load MIL Model -------------------
def load_mil_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Attention().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model.to(device)


# ------------------- Run MIL Inference on One HDF5 -------------------
def run_mil_inference(hdf5_file, model):
    with h5py.File(hdf5_file, 'r') as h5f:
        features = np.array(h5f['embeddings'])
        coords = np.array(h5f['coords'])

    device = next(model.parameters()).device
    input_features = torch.from_numpy(features).float().unsqueeze(0).to(device)
    model.return_attention = True

    with torch.no_grad():
        _, Y_prob, Y_hat, A = model(input_features)

    attention_scores = A.cpu().numpy().squeeze(0)
    preds = Y_hat.cpu().numpy()

    return attention_scores, preds, coords

# ------------------- Batch MIL Inference Over Folder -------------------
def run_inference_on_folder(h5_folder_path, model_path, threshold=0.5):
    # Load MIL model
    model = load_mil_model(model_path)
    print(f" MIL model loaded from: {model_path}")

    # List HDF5 files
    h5_files = [f for f in os.listdir(h5_folder_path) if f.endswith('.hdf5')]
    if not h5_files:
        print(" No HDF5 files found in the folder.")
        return

    print(f"Found {len(h5_files)} HDF5 files for inference.\n")

    # Process each file
    for file in h5_files:
        hdf5_file_path = os.path.join(h5_folder_path, file)
        print(f"▶️ Running inference on: {file}")

        try:
            attention_scores, preds, coords = run_mil_inference(hdf5_file_path, model)

            # Save outputs
            np.save(hdf5_file_path.replace('.hdf5', '_attention_scores.npy'), attention_scores)
            np.save(hdf5_file_path.replace('.hdf5', '_coords.npy'), coords)

            # Display prediction summary
            print(f" Prediction: {preds[0]}, Max Attention: {np.max(attention_scores):.4f}")
            print(f"Results saved for {file}\n")
        
        except Exception as e:
            print(f" Error processing {file}: {e}\n")

    print(" All MIL inferences completed!")

# ------------------- Example Call -------------------
if __name__ == "__main__":
    h5_folder = "/path/to/h5/folder"  # Replace with actual folder path
    model_path = "/path/to/mil/model.pth"  # Replace with actual MIL model path
    run_inference_on_folder(h5_folder, model_path)
