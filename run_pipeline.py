#!/usr/bin/env python3
"""
DREAM x CACHE Target 2035 Drug Discovery Challenge
Unified Pipeline CLI Entrypoint (MMELON late-fusion version).

This script implements our high-performance combinatorial late-fusion 
caching strategy for the MMELON multi-view model:
1. Cleans and deduplicates the raw selection data using Polars.
2. Caches the MMELON multi-view embeddings for the physical building blocks once.
3. Automatically maps and combines building block embeddings on-the-fly inside PyTorch.
4. Trains a fast, highly scalable prediction head (MLP) in PyTorch.
"""

from __future__ import annotations
import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import custom modules
from preprocess_del import (
    deduplicate_selection_parquet,
    BuildingBlockMapper,
    CombinatorialMMELONDataset,
    cache_bb_embeddings
)

# ---------------------------------------------------------------------------
# Multi-Layer Perceptron (MLP) Prediction Head for MMELON Embeddings
# ---------------------------------------------------------------------------

class MMELONCombinatorialMLP(nn.Module):
    """
    A highly scalable, fast multi-layer prediction head that maps combined 
    MMELON building block embeddings directly to a pocket binding affinity score.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DREAM x CACHE Target 2035: MMELON Combinatorial Fine-Tuning CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Paths
    parser.add_argument(
        "--selection-file",
        type=str,
        default="PGK2_selection.parquet",
        help="Path to the raw PGK2_selection.parquet file downloaded from AIRCHECK."
    )
    parser.add_argument(
        "--bb-glob",
        type=str,
        default="OpenDEL-libraries/building_blocks/*.parquet",
        help="Glob pattern pointing to the building block parquet files."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="processed_data",
        help="Directory to save deduplicated files, embeddings, and weights."
    )
    
    # Preprocessing Configurations
    parser.add_argument(
        "--scoring-scheme",
        type=str,
        choices=["tier2"],
        default="tier2",
        help="The scoring scheme to map 3 experimental conditions into a 0-1 target score."
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=250000,
        help="Balanced sample size (all actives + sampled inactives) to train MLP head. Set to 0 for no downsampling (full DEL!)."
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.5,
        help="The continuous score threshold above which a compound is labeled active (1)."
    )
    
    # Model configuration
    parser.add_argument(
        "--base-model",
        type=str,
        default="ibm-research/biomed.sm.mv-te-84m",
        help="HuggingFace repository ID for MMELON."
    )
    
    # Training Configuration
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of epochs to train the fast MLP prediction head."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for training."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for optimization."
    )
    
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    print("=" * 60)
    print("   DREAM x CACHE TARGET 2035 - MMELON PIPELINE CLI")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    dedup_output_path = os.path.join(args.output_dir, "PGK2_selection_deduplicated.parquet")
    bb_embeddings_path = os.path.join(args.output_dir, "mmelon_bb_embeddings.npz")
    model_save_path = os.path.join(args.output_dir, "mmelon_mlp_head.pt")
    
    # Check GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Hardware accelerator: {device.upper()}")
    
    # 1. Clean & Deduplicate raw selection data
    if not os.path.exists(args.selection_file):
        print(f"❌ Error: Raw selection file '{args.selection_file}' not found.")
        print("Please download it from AIRCHECK and place it in the workspace.")
        return
        
    print(f"\n[Step 1/5] Starting structural resolution and deduplication...")
    deduplicate_selection_parquet(
        input_path=args.selection_file,
        output_path=dedup_output_path,
        dedup_col="SMILES",
        compound_col="compound"
    )
    
    # 2. Map Building Block SMILES Lookup
    print(f"\n[Step 2/5] Mapping building block chemical structures...")
    bb_mapper = BuildingBlockMapper(bb_files_glob=args.bb_glob)
    if len(bb_mapper.bb_map) == 0:
        print("⚠️ Warning: No building blocks mapped. Ensure 'OpenDEL-libraries' is downloaded.")
    else:
        print(f"✔ Successfully loaded {len(bb_mapper.bb_map):,} building blocks.")
        
    # 3. Extract and Cache Building Block Embeddings using pre-trained MMELON
    print(f"\n[Step 3/5] Extracting & caching MMELON building block embeddings...")
    cache_bb_embeddings(
        bb_mapper=bb_mapper,
        base_model_path=args.base_model,
        output_path=bb_embeddings_path,
        device=device
    )
    
    # 4. Instantiate combinatorial dataset and data loader
    print(f"\n[Step 4/5] Preparing highly scalable PyTorch dataset...")
    sample_limit = args.sample_size if args.sample_size > 0 else None
    
    try:
        dataset = CombinatorialMMELONDataset(
            selection_parquet_path=dedup_output_path,
            bb_embeddings_path=bb_embeddings_path,
            scoring_scheme=args.scoring_scheme,
            score_threshold_labeling=args.score_threshold,
            sample_size=sample_limit,
        )
        
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4 if device == "cuda" else 0,
            pin_memory=True if device == "cuda" else False
        )
        
        # 5. Train lightweight MLP prediction head
        print(f"\n[Step 5/5] Fine-tuning the fast MLP prediction head...")
        mlp_head = MMELONCombinatorialMLP(input_dim=dataset.emb_dim).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(mlp_head.parameters(), lr=args.lr, weight_decay=1e-4)
        
        print(f"Training for {args.epochs} epochs over {len(dataset):,} samples...")
        for epoch in range(1, args.epochs + 1):
            mlp_head.train()
            total_loss = 0.0
            correct = 0
            total = 0
            
            for embs, labels, _ in loader:
                embs = embs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                logits = mlp_head(embs).squeeze(-1)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item() * embs.size(0)
                preds = (torch.sigmoid(logits) > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
            epoch_loss = total_loss / total
            epoch_acc = (correct / total) * 100.0
            print(f"  Epoch {epoch:02d}/{args.epochs:02d} - Loss: {epoch_loss:.4f} - Train Acc: {epoch_acc:.2f}%")
            
        # Save model weights and configuration
        print(f"\nSaving trained MLP head weights to: {model_save_path}")
        torch.save({
            "state_dict": mlp_head.state_dict(),
            "input_dim": dataset.emb_dim,
            "scoring_scheme": args.scoring_scheme,
            "base_model": args.base_model
        }, model_save_path)
        
        print(f"\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("-" * 50)
        print(f"MMELON Combinatorial late-fusion pipeline ready:")
        print(f"  - Cleaned & Deduplicated parquet saved.")
        print(f"  - Pre-computed building block embeddings cached.")
        print(f"  - Prediction head fully trained & weights saved.")
        print("-" * 50)
        
    except Exception as e:
        print(f"❌ Error during dataset generation/fine-tuning: {e}")


if __name__ == "__main__":
    main()
