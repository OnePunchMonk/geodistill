"""Script to generate teacher annotations offline and cache them."""

import os
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm

from geodistill.config import TeacherConfig
from geodistill.teachers.depth_anything import DepthAnythingTeacher
from geodistill.teachers.vggt import VGGTTeacher
from geodistill.teachers.multi_teacher import MultiTeacherEnsemble


def parse_args():
    parser = argparse.ArgumentParser(description="Build Teacher Dataset")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory with source images")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save annotations")
    parser.add_argument("--teacher", type=str, default="multi", choices=["vggt", "depth_anything", "multi"])
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    
    config = TeacherConfig(teacher_type=args.teacher, device=args.device)
    
    if args.teacher == "vggt":
        teacher = VGGTTeacher(config)
    elif args.teacher == "depth_anything":
        teacher = DepthAnythingTeacher(config)
    else:
        teacher = MultiTeacherEnsemble(config)
        
    os.makedirs(os.path.join(args.output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "depths"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "normals"), exist_ok=True)
    
    image_paths = sorted(list(Path(args.image_dir).glob("*.jpg")) + list(Path(args.image_dir).glob("*.png")))
    
    print(f"Found {len(image_paths)} images. Processing...")
    
    for img_path in tqdm(image_paths):
        image = Image.open(img_path).convert("RGB")
        
        # We need a batch dimension for teacher
        image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(args.device)
        
        with torch.no_grad():
            outputs = teacher(image_tensor)
            
        base_name = img_path.stem
        
        # Save image copy
        image.save(os.path.join(args.output_dir, "images", f"{base_name}.jpg"))
        
        if "depth" in outputs:
            depth = outputs["depth"].squeeze().cpu().numpy()
            np.save(os.path.join(args.output_dir, "depths", f"{base_name}.npy"), depth)
            
        if "normal" in outputs:
            normal = outputs["normal"].squeeze().cpu().numpy()
            np.save(os.path.join(args.output_dir, "normals", f"{base_name}.npy"), normal)

if __name__ == "__main__":
    main()
