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
