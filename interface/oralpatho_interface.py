import ipywidgets as widgets
from IPython.display import display, clear_output, HTML
import os
import h5py
import numpy as np

from combined_wsi_patch_extractor_feature import process_slide_and_extract_features_individual
from inference import load_mil_model, run_mil_inference
from display_attention_on_wsi import display_attention_on_wsi
#from visualization import display_attention_on_wsi

# ---------------- Configuration ----------------
WSI_FOLDER = "/path/to/wsi/"
HDF5_OUTPUT_FOLDER = "/path/to/features_out/"

# ---------------- Dynamic File Lists -----------------
def refresh_file_lists():
    wsi_files = [f for f in os.listdir(WSI_FOLDER) if f.endswith(('.svs', '.ndpi'))]
    h5_dirs = [os.path.join(HDF5_OUTPUT_FOLDER, d) for d in os.listdir(HDF5_OUTPUT_FOLDER) if os.path.isdir(os.path.join(HDF5_OUTPUT_FOLDER, d))]
    return wsi_files, h5_dirs

# ----------------- Tab 1: Combined Patch and Feature Extraction ------------------
def tab_generate_patches(wsi_files):
    wsi_selector = widgets.Combobox(options=wsi_files, description='Select WSI:', placeholder='Or enter WSI name')
    patch_size_slider = widgets.IntSlider(value=256, min=128, max=512, step=128, description='Patch Size:')
    threshold_slider = widgets.FloatSlider(value=0.75, min=0.1, max=0.9, step=0.05, description='Tissue Threshold:')
    run_button = widgets.Button(description='Run Patch + Feature', button_style='success')
    output = widgets.Output()

    def run_combined_pipeline(b):
        with output:
            clear_output(wait=True)
            selected_wsi = os.path.join(WSI_FOLDER, wsi_selector.value)
            print(f"Selected WSI: {selected_wsi}")
            slide_name = os.path.splitext(wsi_selector.value)[0]
            hdf5_save_path = os.path.join(HDF5_OUTPUT_FOLDER, f'{slide_name}_features_resnet50.hdf5')
            process_slide_and_extract_features_individual(
                filename=selected_wsi,
                tile_size=patch_size_slider.value,
                tissue_threshold=threshold_slider.value,
                h5_output_path=hdf5_save_path
            )
            print("✅ Combined patch and feature extraction completed.")

    run_button.on_click(run_combined_pipeline)
    return widgets.VBox([wsi_selector, patch_size_slider, threshold_slider, run_button, output])

# ----------------- Tab 2: MIL Inference ------------------
def tab_mil_inference(h5_dirs):
    h5_dir_selector = widgets.Combobox(options=h5_dirs, description='Select HDF5 Folder:', placeholder='Or enter folder path')
    model_selector = widgets.Text(description='Model Path:', placeholder='Enter MIL model path')
    run_button = widgets.Button(description='Run MIL Inference', button_style='info')
    output = widgets.Output()

    def run_mil(b):
        with output:
            clear_output(wait=True)
            model = load_mil_model(model_selector.value)
            print(f"✅ MIL Model Loaded: {model_selector.value}")
            for file in os.listdir(h5_dir_selector.value):
                if file.endswith('.hdf5'):
                    hdf5_file_path = os.path.join(h5_dir_selector.value, file)
                    scores, preds, coords = run_mil_inference(hdf5_file_path, model)
                    np.save(hdf5_file_path.replace('.hdf5', '_attention_scores.npy'), scores)
                    np.save(hdf5_file_path.replace('.hdf5', '_coords.npy'), coords)
                    print(f"✅ MIL Inference completed for {file}. Scores and coords saved.")

    run_button.on_click(run_mil)
    return widgets.VBox([h5_dir_selector, model_selector, run_button, output])

# ----------------- Tab 3: Visualization ------------------
def tab_visualization(wsi_files, h5_dirs):
    wsi_selector = widgets.Combobox(options=wsi_files, description='Select WSI:', placeholder='Or enter WSI name')
    h5_dir_selector = widgets.Combobox(options=h5_dirs, description='Select HDF5 Folder:', placeholder='Or enter folder path')
    attention_slider = widgets.FloatSlider(value=0.5, min=0.1, max=1.0, step=0.05, description='Attention Threshold:')
    run_button = widgets.Button(description='Visualize', button_style='warning')
    output = widgets.Output()

    def visualize(b):
        with output:
            clear_output(wait=True)
            for file in os.listdir(h5_dir_selector.value):
                if file.endswith('.hdf5'):
                    hdf5_file_path = os.path.join(h5_dir_selector.value, file)
                    coords = np.load(hdf5_file_path.replace('.hdf5', '_coords.npy'))
                    scores = np.load(hdf5_file_path.replace('.hdf5', '_attention_scores.npy'))
                    display_attention_on_wsi(os.path.join(WSI_FOLDER, wsi_selector.value), coords, scores, threshold=attention_slider.value)

    run_button.on_click(visualize)
    return widgets.VBox([wsi_selector, h5_dir_selector, attention_slider, run_button, output])

# ----------------- Combine Tabs ------------------
def create_interface():
    wsi_files, h5_dirs = refresh_file_lists()
    
    title_html = HTML("""
    <div style='text-align: center; padding: 20px; border: 2px solid #4CAF50; border-radius: 15px; background-color: #f9f9f9;'>
        <h1 style='color: #4CAF50; font-family: Arial, sans-serif;'>ORALPATHO</h1>
        <h3 style='color: #333;'>Interactive AI Interface for Oral Cancer Histopathology</h3>
        <p style='font-size: 16px; color: #555;'>
            Welcome to <strong>ORALPATHO</strong> — a <em>modular and interactive AI pipeline</em> designed for whole slide image analysis in oral cancer research and diagnosis.
        </p>
        <ul style='list-style: none; padding: 0; font-size: 15px;'>
            <li>🔹 Patch extraction and feature computation.</li>
            <li>🔹 MIL model inference for cancer detection.</li>
            <li>🔹 Interactive visualization of AI-generated heatmaps over WSIs.</li>
        </ul>
        <p style='font-size: 16px; color: #333;'><strong>Get Started:</strong> Use the tabs below to navigate through each step of the pipeline.</p>
    </div>
    """)

    tabs = widgets.Tab()
    tabs.children = [
        tab_generate_patches(wsi_files),
        tab_mil_inference(h5_dirs),
        tab_visualization(wsi_files, h5_dirs)
    ]
    tabs.set_title(0, 'Generate Patches + Features')
    tabs.set_title(1, 'Run MIL Inference')
    tabs.set_title(2, 'Visualization')

    display(title_html, tabs)

# Run the interface
create_interface()