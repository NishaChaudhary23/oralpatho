
# OralPatho

**OralPatho** is a dual-stage AI system for pathologist-free tumor detection and histological grading of Oral Squamous Cell Carcinoma (OSCC) from whole-slide histopathology images (WSIs). Trained on over 1900 WSIs from multiple institutions, it performs both binary cancer detection and multi-class grading (WD, MD, PD) with expert-level accuracy and visual interpretability.


## 🔍 Key Features

- **Dual-stage MIL framework** for normal-vs-tumor classification and tumor grading
- **Attention heatmaps** for explainability at patch level
- **Preprocessing notebooks** for patch extraction, normalization, and metadata mapping
- **Plug-and-play WSI analysis** with minimal annotation requirements


## 📁 Repository Structure
```
oralpatho/
├── data/                # Instructions or placeholders for WSI datasets
├── docs/                # Documentation and figure panels
├── examples/            # Inference examples and CLI usage
├── results/             # Saved model outputs and evaluation
├── scripts/             # Command-line scripts for full pipeline
├── src/                 # Core source code
│   ├── preprocessing/   # Jupyter notebooks for patch extraction, normalization
│   ├── training/        # Training code for binary & multiclass MIL models
│   └── utils/           # Utilities (logger, metrics, model definitions)
├── LICENSE
├── README.md
├── requirements.txt     # Python dependencies
```

---

## ⚙️ Installation
```bash
# Clone the repo
$ git clone https://github.com/NishaChaudhary23/oralpatho.git
$ cd oralpatho

# Create environment (optional)
$ python3 -m venv venv
$ source venv/bin/activate

# Install dependencies
$ pip install -r requirements.txt
```

---

## 🚀 Quick Start
1. Place WSIs and metadata in the `data/` directory.
2. Run preprocessing notebooks under `src/preprocessing/`
3. Train models:
   - Binary: `src/training/binary/`
   - Multiclass: `src/training/multiclass/`
4. Visualize attention heatmaps and predictions from saved outputs in `results/`

---

## 📄 Citation
If you use OralPatho in your research, please cite our preprint:

**Chaudhary N, et al.** *OralPatho: Pathologist-Free Dual-Stage AI System for Tumor Detection and Grading in Oral Squamous Cell Carcinoma*. medRxiv 2025. https://doi.org/10.1101/2025.06.05.25329090

---

## 👩‍⚕️ Contact
Lead developer: **Nisha Chaudhary**  
Email: [nickychaudhary23@gmail.com](mailto:nickychaudhary23@gmail.com)

Project mentor: **Dr. Tanveer Ahmad**  
Institution: Jamia Millia Islamia, New Delhi

---

> This project aims to enable scalable and interpretable cancer diagnostics in clinical and resource-limited settings.
=======
# oralpatho
## Overview

oralpatho is a dual-stage AI pipeline designed for automated detection and histological subtyping of Oral Squamous Cell Carcinoma (OSCC) from whole-slide histopathology images. 

It leverages an attention-based multiple instance learning (MIL) framework:
- **Stage 1**: Binary classification (OSCC vs. Normal)
- **Stage 2**: Multiclass OSCC grading (Well, Moderate, Poor)

The model operates without patch-level annotations, using weak supervision to localise tumour regions and assign subtype labels with high accuracy.

This repository includes:
- Code for data preprocessing and patch extraction
- Feature extraction using ResNet50
- Training scripts for both binary and multiclass MIL models
- Attention heatmap visualisation and evaluation workflows
>>>>>>> 02a9fb1f5f4cf9519f56b5d649f6b1d0b1d9a8a9
