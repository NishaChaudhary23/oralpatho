import openslide
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from skimage.feature import canny
from skimage.morphology import binary_closing, binary_dilation, disk
from scipy.ndimage import binary_fill_holes
import h5py
import time
from PIL import Image
from tqdm import tqdm

# ------------------- Device Configuration -------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device in use:", device)

# ------------------- Model Setup -------------------
model = torchvision.models.resnet50(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
    
for param in list(model.parameters())[-30:]:
    param.requires_grad = True

model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))     

model.fc = nn.Sequential(
    nn.Linear(2048, 512),
    nn.ReLU(inplace=True),
)
num_ftrs = 512

class fully_connected(nn.Module):
    def __init__(self, model, num_ftrs, num_classes):
        super(fully_connected, self).__init__()
        self.model = model
        self.fc_4 = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        x = self.model(x)
        out_1 = x
        out_3 = self.fc_4(x)
        return out_1, out_3

model_final = fully_connected(model, num_ftrs, 4).to(device)
model_final = nn.DataParallel(model_final)
model_final.eval()

# ------------------- Transform -------------------
trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ------------------- Helper Functions -------------------
def open_slide(filename):
    try:
        return openslide.OpenSlide(filename)
    except Exception as e:
        print(f"Error opening slide: {e}")
        return None

def keep_tile(tile, tissue_threshold):
    edges = np.zeros(tile.shape[:2], dtype=bool)
    for channel in range(tile.shape[2]):
        edges |= canny(tile[:, :, channel])
    tile_closed = binary_closing(edges, disk(10))
    tile_dilated = binary_dilation(tile_closed, disk(10))
    tile_filled = binary_fill_holes(tile_dilated)
    return tile_filled.mean() >= tissue_threshold

# ------------------- Main Patch + Feature Extraction (No quick skip, no low-res detection) -------------------
def process_slide_and_extract_features_individual(filename, tile_size, tissue_threshold, h5_output_path):
    slide = open_slide(filename)
    if slide is None:
        return

    slide_name = os.path.splitext(os.path.basename(filename))[0]
    highest_res_level = 0  # Always use highest resolution for tissue detection
    wsi_width, wsi_height = slide.level_dimensions[highest_res_level]
    print(f"Processing {slide_name}: {wsi_width}x{wsi_height}")

    embeddings_list = []
    coordinates_list = []
    slide_ids = []

    # Iterate over the whole slide.
    # 'col' corresponds to the x-coordinate (width) and 'row' corresponds to the y-coordinate (height).
    for col in tqdm(range(0, wsi_width, tile_size), desc="Columns"):
        for row in range(0, wsi_height, tile_size):
            # Read a patch from the slide and convert to RGB.
            tile_image = slide.read_region((col, row), highest_res_level, (tile_size, tile_size)).convert("RGB")
            tile_np = np.array(tile_image)

            # Check if the patch contains sufficient tissue.
            if keep_tile(tile_np, tissue_threshold):
                # Transform the patch and add a batch dimension.
                patch_tensor = trans(tile_image).unsqueeze(0).to(device)
                with torch.no_grad():
                    output_features, _ = model_final(patch_tensor)
                embedding = output_features.cpu().numpy().squeeze(0)

                # Store the embedding and the coordinate (row, col).
                embeddings_list.append(embedding)
                coordinates_list.append((row, col))
                slide_ids.append(slide_name)

    # Save to HDF5
    with h5py.File(h5_output_path, 'w') as h5_file:
        h5_file.create_dataset('embeddings', data=np.array(embeddings_list), dtype='float32')
        h5_file.create_dataset('coords', data=np.array(coordinates_list), dtype='int32')
        h5_file.attrs['slide_id'] = slide_ids

    print(f"Finished processing {slide_name}: Total patches saved: {len(coords_list)}")

# ------------------- Example Call -------------------
if __name__ == "__main__":
    input_wsi_path = "/media/microcrispr8/DATA 1/oscc-tcia/sample_wsi.svs"  
    h5_output_path = "/media/microcrispr8/DATA 1/DSMIL/features/features_from_resnet50_oscc_tcia/sample_wsi_features.h5"
    process_slide_and_extract_features(
        filename=input_wsi_path,
        tile_size=256,
        tissue_threshold=0.75,
        h5_output_path=h5_output_path,
        batch_size=1500  
    )
    
    

