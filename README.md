# GeoDistill: Geometry-Grounded Distillation

GeoDistill is a robust distillation framework that transfers 3D geometric knowledge from large foundation models (like VGGT and Depth Anything V2) into lightweight ViT students equipped with LoRA adapters.

## Overview

Unlike standard knowledge distillation that targets logits or intermediate features, GeoDistill targets a structured geometric space (metric depth, surface normals, point maps). This forces the student to build an implicit 3D world model, producing superior representations for downstream dense prediction tasks (detection, segmentation).

## Core Components

- **Multi-Teacher Ensemble**: Combines predictions from multiple state-of-the-art 3D models with uncertainty weighting.
- **LoRA-Equipped Student**: Injects low-rank adapters into a frozen ViT backbone (e.g., DINOv2) to learn geometric features efficiently.
- **DPT Decoder Head**: A lightweight convolutional head for predicting dense outputs.
- **Geometric Losses**: Enforces structural consistency, such as ensuring predicted normals are orthogonal to the depth gradient.

## Usage

1. **Build the offline teacher dataset:**
   ```bash
   python experiments/build_teacher_dataset.py --dataset nyuv2
   ```

2. **Train the student model:**
   ```bash
   accelerate launch src/geodistill/train.py --config experiments/default_config.yaml
   ```

## Next Steps to Continue Building

- [ ] **Multi-Teacher Integration**: Fully integrate MASt3R as an optional third teacher for dense correspondence targets.
- [ ] **Expanded Benchmarks**: Add COCO Object Detection evaluation via ViTDet head.
- [ ] **Hyperparameter Tuning**: Perform a grid search on the loss weights ($\lambda_1$ to $\lambda_6$) to find the optimal balance between geometric consistency and raw depth MSE.
- [ ] **Model Scaling**: Evaluate the distillation pipeline using larger student backbones like ViT-Base or ViT-Large.
