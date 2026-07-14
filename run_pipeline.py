#!/usr/bin/env python3
"""
DREAM x CACHE Target 2035 Drug Discovery Challenge
Unified Pipeline CLI Entrypoint.

This script allows you or your collaborator to run the entire data engineering 
and pre-processing pipeline from the command line. It cleans, deduplicates, 
computes binding scores under different tiers, and instantiates the tokenized PyTorch Dataset.
"""

from __future__ import annotations
import os
import argparse
from pathlib import Path
import pandas as pd
import polars as pl

# Import custom modules
from preprocess_del import deduplicate_selection_parquet, BuildingBlockMapper, LargeMAMMALDataset, load_pgk2_sequence

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DREAM x CACHE Target 2035: MAMMAL Preprocessing & Labeling Pipeline CLI",
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
        "--fasta-file",
        type=str,
        default="pgk2_sequence.fasta",
        help="Path to the FASTA file containing PGK2 amino acid sequence."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="processed_data",
        help="Directory to save deduplicated files and training logs."
    )
    
    # Preprocessing Configurations
    parser.add_argument(
        "--scoring-scheme",
        type=str,
        choices=["tier1", "tier2", "tier3"],
        default="tier2",
        help="The scoring scheme to map 3 experimental conditions into a 0-1 target score."
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=250000,
        help="Balanced sample size (all actives + sampled inactives) to train MAMMAL efficiently. Set to 0 for no downsampling."
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
        default="ibm/biomed.omics.bl.sm.ma-ted-458m",
        help="HuggingFace repository ID for MAMMAL to load tokenizer."
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Use a compact combinatorial prompt style by feeding protein sequence and BB SMILES once, then using BB codes."
    )
    
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    print("=" * 60)
    print("   DREAM x CACHE TARGET 2035 - MAMMAL PIPELINE CLI")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    dedup_output_path = os.path.join(args.output_dir, "PGK2_selection_deduplicated.parquet")
    
    # 1. Clean & Deduplicate raw selection data
    if not os.path.exists(args.selection_file):
        print(f"❌ Error: Raw selection file '{args.selection_file}' not found.")
        print("Please download it from AIRCHECK and place it in the workspace.")
        return
        
    print(f"\n[Step 1/4] Starting structural resolution and deduplication...")
    deduplicate_selection_parquet(
        input_path=args.selection_file,
        output_path=dedup_output_path,
        dedup_col="SMILES",
        compound_col="compound"
    )
    
    # 2. Map Building Block SMILES Lookup
    print(f"\n[Step 2/4] Mapping building block chemical structures...")
    bb_mapper = BuildingBlockMapper(bb_files_glob=args.bb_glob)
    if len(bb_mapper.bb_map) == 0:
        print("⚠️ Warning: No building blocks mapped. Ensure 'OpenDEL-libraries' is downloaded.")
    else:
        print(f"✔ Successfully loaded {len(bb_mapper.bb_map):,} building blocks.")
        
    # 3. Load Target Protein Sequence
    print(f"\n[Step 3/4] Loading PGK2 protein sequence context...")
    pgk2_sequence = load_pgk2_sequence(args.fasta_file)
    print(f"✔ Loaded sequence: {pgk2_sequence[:40]}... ({len(pgk2_sequence)} AA)")
    
    # 4. Tokenize & Verify Dataset Shapes with MAMMAL Tokenizer
    print(f"\n[Step 4/4] Validating tokenization pipeline & PyTorch Dataset...")
    try:
        from fuse.data.tokenizers.modular_tokenizer.op import ModularTokenizerOp
        print(f"Loading MAMMAL Modular Tokenizer from HuggingFace ({args.base_model})...")
        tokenizer_op = ModularTokenizerOp.from_pretrained(args.base_model)
    except ImportError:
        print("❌ Error: 'biomed-multi-alignment' or 'fuse' packages are not installed.")
        print("Please activate your conda environment and run: pip install biomed-multi-alignment")
        return
    except Exception as e:
        print(f"❌ Error loading tokenizer: {e}")
        return
        
    sample_limit = args.sample_size if args.sample_size > 0 else None
    
    print(f"Initializing LargeMAMMALDataset (Scheme: {args.scoring_scheme}, Sample Limit: {sample_limit}, Compact: {args.compact})...")
    try:
        dataset = LargeMAMMALDataset(
            selection_parquet_path=dedup_output_path,
            bb_mapper=bb_mapper,
            protein_sequence=pgk2_sequence,
            tokenizer_op=tokenizer_op,
            scoring_scheme=args.scoring_scheme,
            score_threshold_labeling=args.score_threshold,
            sample_size=sample_limit,
            compact=args.compact,
        )
        print("✔ Dataset loaded successfully.")
        
        # Pull a single sample to verify shapes
        sample = dataset[0]
        from mammal.keys import ENCODER_INPUTS_TOKENS, DECODER_INPUTS_TOKENS
        print(f"\n🎉 PIPELINE SUCCESSFUL AND VERIFIED!")
        print("-" * 50)
        print(f"Verified dataset dimensions for training:")
        print(f"  - Total samples available: {len(dataset):,}")
        print(f"  - Encoder input shape:     {sample[ENCODER_INPUTS_TOKENS].shape} (tokens)")
        print(f"  - Decoder input shape:     {sample[DECODER_INPUTS_TOKENS].shape} (tokens)")
        print(f"  - First sample score:      {sample['data.target_score']:.4f}")
        print(f"  - First sample label:      {sample['data.label']} ({'Active' if sample['data.label'] == 1 else 'Inactive'})")
        print("-" * 50)
        print(f"Preprocessed parquet saved at: {dedup_output_path}")
        print(f"The environment is fully configured and ready for model training!")
        
    except Exception as e:
        print(f"❌ Error during dataset generation: {e}")


if __name__ == "__main__":
    main()
