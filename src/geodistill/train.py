"""Training loop for GeoDistill with Accelerate."""

import os
import argparse
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm

from geodistill.config import GeoDistillConfig
from geodistill.student import GeoDistillStudent
from geodistill.decoder import DPTDecoder
from geodistill.losses import GeoDistillLoss
from geodistill.data import TeacherDataset, GeoTransforms
from geodistill.eval import evaluate_transfer


def parse_args():
    parser = argparse.ArgumentParser(description="Train GeoDistill")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    return parser.parse_args()


def main():
    args = parse_args()
    config = GeoDistillConfig.from_yaml(args.config)
    
    accelerator = Accelerator(
        mixed_precision=config.training.mixed_precision,
        log_with="wandb" if config.training.wandb_project else None
    )
    
    if accelerator.is_main_process:
        os.makedirs(config.training.output_dir, exist_ok=True)
        if config.training.wandb_project:
            accelerator.init_trackers(
                project_name=config.training.wandb_project,
                config=config.__dict__
            )

    # Models
    student = GeoDistillStudent(config.student)
    decoder = DPTDecoder(config.decoder, config.student)
    criterion = GeoDistillLoss(config.loss)
    
    # Optimizer
    params = list(filter(lambda p: p.requires_grad, student.parameters())) + \
             list(decoder.parameters())
             
    if criterion.log_vars is not None:
        params.append(criterion.log_vars)
             
    optimizer = optim.AdamW(
        params, 
        lr=config.training.learning_rate, 
        weight_decay=config.training.weight_decay
    )
    
    # Scheduler (Cosine with warmup)
    # Using a simple one for illustration
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config.training.max_epochs
    )

    # Dataset & DataLoader
    transform = GeoTransforms(config.data.image_size, augment=config.data.augment)
    
    # Assume teacher dataset holds the ground truths
    train_dataset = TeacherDataset(
        config.data.teacher_cache_dir, 
        split=config.data.train_split, 
        transform=transform
    )
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config.data.batch_size, 
        shuffle=True, 
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory
    )
    
    # Accelerate prepare
    student, decoder, criterion, optimizer, train_dataloader, scheduler = accelerator.prepare(
        student, decoder, criterion, optimizer, train_dataloader, scheduler
    )
    
    global_step = 0
    
    for epoch in range(config.training.max_epochs):
        student.train()
        decoder.train()
        
        pbar = tqdm(train_dataloader, disable=not accelerator.is_main_process)
        for batch in pbar:
            images = batch["image"]
            
            # Ground truth targets from teacher dataset
            targets = {}
            if "depth" in batch:
                targets["depth"] = batch["depth"]
            if "normal" in batch:
                targets["normal"] = batch["normal"]
                
            with accelerator.accumulate(student):
                outputs = student(images)
                preds = decoder(outputs["features"], (images.shape[2], images.shape[3]))
                
                # Combine outputs (student features + decoder predictions)
                student_preds = {**outputs, **preds}
                
                # Assume teacher dataset doesn't have features stored directly, we might need a teacher wrapper
                # For this script, we just assume targets contains everything needed.
                losses = criterion(student_preds, targets, images=images)
                loss = losses["loss"]
                
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params, config.training.gradient_clip_norm)
                    
                optimizer.step()
                optimizer.zero_grad()
                
            global_step += 1
            
            if accelerator.is_main_process:
                pbar.set_postfix({"loss": loss.item()})
                if global_step % config.training.log_interval == 0:
                    accelerator.log({"train/loss": loss.item(), "train/lr": scheduler.get_last_lr()[0]}, step=global_step)
                    
        scheduler.step()
        
        # Save checkpoint
        if accelerator.is_main_process and (epoch + 1) % config.training.save_interval == 0:
            checkpoint_path = Path(config.training.output_dir) / f"checkpoint-{epoch+1}"
            accelerator.save_state(checkpoint_path)

    if accelerator.is_main_process:
        accelerator.end_training()


if __name__ == "__main__":
    main()
