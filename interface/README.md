# ORALPATHO Interface

This module provides an interactive interface for running the ORALPATHO pipeline on Whole Slide Images (WSIs). It supports:

- Patch extraction and feature computation  
- MIL model inference  
- Attention heatmap visualization  

## How to Run

Run the script in a Jupyter Notebook environment:

    cd interface
    jupyter notebook oralpatho_interface.py

Make sure all dependencies from the main requirements.txt are installed.

## Features

1. Patch + Feature Extraction: Run tile extraction and ResNet50 feature saving in one step.  
2. MIL Inference: Load your MIL model and get predictions + attention scores.  
3. Visualization: Generate attention-based heatmaps on WSIs.

Dependencies include: ipywidgets, openslide-python, histomicstk, numpy, h5py, Pillow, matplotlib
