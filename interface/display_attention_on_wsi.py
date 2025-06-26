import openslide
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from histomicstk.preprocessing.color_normalization import reinhard
from PIL import Image
import os

# === Color Normalization Params ===
cnorm = {
    'mu': np.array([8.74108109, -0.12440419, 0.0444982]),
    'sigma': np.array([0.6135447, 0.10989545, 0.0286032]),
}

def normalize_wsi(wsi_image):
    img_array = np.array(wsi_image.convert("RGB"))
    normalized_img = reinhard(img_array, target_mu=cnorm['mu'], target_sigma=cnorm['sigma'])
    normalized_img = np.clip(normalized_img, 0, 255).astype(np.uint8)
    return Image.fromarray(normalized_img)

def display_attention_on_wsi(wsi_path, coords, scores, threshold=0.5, tile_size=256, display_level=2):
    slide_id = os.path.splitext(os.path.basename(wsi_path))[0]
    slide = openslide.OpenSlide(wsi_path)
    display_level = min(display_level, slide.level_count - 1)
    downsampled = slide.read_region((0, 0), display_level, slide.level_dimensions[display_level])

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(downsampled)
    ax.set_title(f"Attention Heatmap – {slide_id}", fontsize=18)

    scale = slide.level_downsamples[display_level]
    norm = mcolors.Normalize(vmin=-0.1, vmax=max(scores))
    cmap = cm.get_cmap("jet")

    for (x_raw, y_raw), score in zip(coords, scores):
        if score > threshold:
            x = x_raw * tile_size / scale
            y = y_raw * tile_size / scale
            w = h = tile_size / scale
            color = cmap(norm(score))
            rect = patches.Rectangle((x, y), w, h, linewidth=0, facecolor=color, alpha=0.7)
            ax.add_patch(rect)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Attention Score")
    plt.show()
