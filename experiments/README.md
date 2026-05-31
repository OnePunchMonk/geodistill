# GeoDistill Experiments

This directory contains scripts and configurations for training and evaluating GeoDistill models.

## Pipeline Steps

1. **Build Teacher Dataset**
   Run the teacher models offline to generate ground truth depth and normal maps.
   ```bash
   python build_teacher_dataset.py \
       --image_dir /path/to/raw/images \
       --output_dir ../data/teacher_cache/train \
       --teacher multi
   ```

2. **Train Student**
   Train the ViT-based student model using the offline teacher cache.
   ```bash
   accelerate launch ../src/geodistill/train.py --config default_config.yaml
   ```

3. **Evaluate Transfer**
   Use `eval` utilities to measure zero-shot transfer on downstream datasets (NYUv2, ScanNet).
