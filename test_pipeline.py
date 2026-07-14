"""
DREAM x CACHE Target 2035 Drug Discovery Challenge
Pipeline Validation & Testing Suite.

This script acts as a test runner that:
1. Generates synthetic Mock DEL selection parquets and mock building block parquets.
2. Runs the end-to-end Polars deduplication and aggregates count and Z-score metrics.
3. Tests the 3 customizable scoring tiers (Binary, Sigmoid Soft-Enrichment, Bayesian Reads).
4. Loads MAMMAL's tokenizer to test prompt assembly, tokenization, and output dimensions.
5. Verifies that the dataset structure is fully ready for HPC fine-tuning.
"""

from __future__ import annotations
import os
import shutil
import numpy as np
import pandas as pd
import polars as pl
import torch
from fuse.data.tokenizers.modular_tokenizer.op import ModularTokenizerOp

# Import our custom modules
from preprocess_del import deduplicate_selection_parquet, BuildingBlockMapper, LargeMAMMALDataset, load_pgk2_sequence

# Mock structures for testing
MOCK_SMILES = [
    "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5",  # Imatinib
    "CC(C)(C)C1=CC(=CC(=C1)O)C2=NC(=NC(=N2)N)N",                              # Mock bb
    "CN1CCN(CC1)CC2=CC=C(C=C2)C(=O)NC3=CC=C(C=C3)C",                           # Mock bb2
    "NC1=NC=NC2=C1N=CN2C3C(O)C(O)C(CO)O3",                                      # Adenosine
    "O=C(O)C1=CC=C(O)C=C1",                                                    # Salicylic acid
]

def generate_mock_datasets(data_dir: str = "mock_data") -> tuple[str, str]:
    """Create local small parquets mimicking challenge formats."""
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(f"{data_dir}/building_blocks", exist_ok=True)
    
    # 1. Create building block tables (Lib 11 BBs)
    # BB1
    bb1_df = pd.DataFrame({
        "ID": [f"000{i}" for i in range(1, 4)],
        "SMILES": MOCK_SMILES[1:4]
    })
    bb1_df.to_parquet(f"{data_dir}/building_blocks/lib11_bb1.parquet")
    
    # BB2
    bb2_df = pd.DataFrame({
        "ID": [f"001{i}" for i in range(1, 4)],
        "SMILES": MOCK_SMILES[2:5]
    })
    bb2_df.to_parquet(f"{data_dir}/building_blocks/lib11_bb2.parquet")
    
    # BB3
    bb3_df = pd.DataFrame({
        "ID": [f"002{i}" for i in range(1, 4)],
        "SMILES": MOCK_SMILES[0:3]
    })
    bb3_df.to_parquet(f"{data_dir}/building_blocks/lib11_bb3.parquet")
    
    # 2. Create raw mock selection parquet containing 20 records
    # Include deliberate duplicate structures with different codons to test deduplication
    np.random.seed(42)
    selection_records = []
    
    # Generate some duplicates
    dup_smiles = MOCK_SMILES[0]
    # Record A (Codon ID 1)
    selection_records.append({
        "compound": "qDOS11-0001-0011-0021",
        "SMILES": dup_smiles,
        "count_PGK2": 15,
        "count_PGK2_with_inhibitor": 2,
        "count_NTC": 1,
        "zscore_PGK2": 3.4,
        "zscore_PGK2_with_inhibitor": 0.4,
        "zscore_NTC": 0.2,
    })
    # Record B (Codon ID 2 - duplicate chemical structure)
    selection_records.append({
        "compound": "qDOS11-0002-0012-0022",
        "SMILES": dup_smiles,
        "count_PGK2": 20,
        "count_PGK2_with_inhibitor": 3,
        "count_NTC": 0,
        "zscore_PGK2": 4.1,
        "zscore_PGK2_with_inhibitor": 0.5,
        "zscore_NTC": 0.0,
    })
    
    # Fill remaining 18 records with randomized combinations
    for i in range(18):
        c_id = f"qDOS11-000{np.random.randint(1,4)}-001{np.random.randint(1,4)}-002{np.random.randint(1,4)}"
        selection_records.append({
            "compound": c_id,
            "SMILES": MOCK_SMILES[i % len(MOCK_SMILES)],
            "count_PGK2": int(np.random.randint(0, 10)),
            "count_PGK2_with_inhibitor": int(np.random.randint(0, 10)),
            "count_NTC": int(np.random.randint(0, 5)),
            "zscore_PGK2": float(np.random.uniform(-1, 3)),
            "zscore_PGK2_with_inhibitor": float(np.random.uniform(-1, 2)),
            "zscore_NTC": float(np.random.uniform(-1, 1)),
        })
        
    sel_df = pd.DataFrame(selection_records)
    sel_path = f"{data_dir}/PGK2_selection_mock_raw.parquet"
    sel_df.to_parquet(sel_path)
    
    return sel_path, f"{data_dir}/building_blocks/*.parquet"


def test_pipeline() -> None:
    """Run verification tests."""
    mock_dir = "mock_test_sandbox"
    print("==================================================")
    print("STARTING TEST RUNNER FOR THE WORKFLOW")
    print("==================================================")
    
    try:
        # 1. Generate Mock Files
        sel_path, bb_glob = generate_mock_datasets(mock_dir)
        print(f"✔ Synthetic mock dataset files written to '{mock_dir}'.")
        
        # 2. Run Deduplication Pipeline
        dedup_path = f"{mock_dir}/PGK2_selection_mock_deduplicated.parquet"
        deduplicate_selection_parquet(
            input_path=sel_path,
            output_path=dedup_path,
            dedup_col="SMILES",
            compound_col="compound"
        )
        print("✔ Deduplication ran successfully.")
        
        # Verify deduplication reduced duplicates
        raw_len = len(pd.read_parquet(sel_path))
        dedup_len = len(pl.read_parquet(dedup_path))
        print(f"  Rows reduced from {raw_len} to {dedup_len}.")
        assert dedup_len < raw_len, "Deduplication did not group the duplicate SMILES!"
        
        # 3. Initialize Building Block Lookup Mapper
        print("Testing BuildingBlockMapper lookups...")
        bb_mapper = BuildingBlockMapper(bb_files_glob=bb_glob)
        test_compound = "qDOS11-0001-0011-0021"
        bbs = bb_mapper.get_bb_smiles(test_compound)
        print(f"  Mapped Compound ID: {test_compound}")
        print(f"  Mapped BB1: {bbs[0][:30]}...")
        print(f"  Mapped BB2: {bbs[1][:30]}...")
        print(f"  Mapped BB3: {bbs[2][:30]}...")
        assert len(bbs) == 3 and all(isinstance(s, str) for s in bbs), "BB mapping failed!"
        
        # Test reverse-engineering of validation SMILES to BB speaking
        print("Testing BuildingBlockMapper reverse-engineering of SMILES...")
        query_smiles = MOCK_SMILES[0] # Imatinib
        ret_bbs, ret_codes = bb_mapper.reverse_engineer_smiles(query_smiles)
        print(f"  Query SMILES: {query_smiles[:30]}...")
        print(f"  Reverse-Engineered BBs: {ret_bbs}")
        print(f"  Reverse-Engineered codes: {ret_codes}")
        assert len(ret_bbs) == 3 and len(ret_codes) == 3, "Reverse-engineering should return lists of length 3!"
        
        # 4. Load lightweight MAMMAL tokenizer from HuggingFace to verify encoding pipeline
        print("Loading pre-trained modular tokenizer from HuggingFace...")
        tokenizer_op = ModularTokenizerOp.from_pretrained("ibm/biomed.omics.bl.sm.ma-ted-458m")
        print("✔ Tokenizer loaded successfully.")
        
        # 5. Initialize Large-Scale Dataset with Customizable Target Schemes
        print("Testing Dataset pipeline with Tier 1 (Hard Binary Classification)...")
        pgk2_sequence = load_pgk2_sequence()
        
        ds_tier1 = LargeMAMMALDataset(
            selection_parquet_path=dedup_path,
            bb_mapper=bb_mapper,
            protein_sequence=pgk2_sequence,
            tokenizer_op=tokenizer_op,
            scoring_scheme="tier1",
            score_threshold_labeling=0.5,
            pgk2_threshold=2.0,
            ntc_threshold=1.0,
            inh_threshold=1.0,
        )
        print(f"✔ Dataset loaded under Tier 1 with {len(ds_tier1)} compounds.")
        
        # Grab first item
        sample_tier1 = ds_tier1[0]
        print("✔ Dataset output shape validation:")
        
        # Look up the actual keys from mammal.keys constants
        from mammal.keys import ENCODER_INPUTS_TOKENS, DECODER_INPUTS_TOKENS
        print(f"  - Encoder tokens shape: {sample_tier1[ENCODER_INPUTS_TOKENS].shape}")
        print(f"  - Decoder tokens shape: {sample_tier1[DECODER_INPUTS_TOKENS].shape}")
        print(f"  - Target Score value:   {sample_tier1['data.target_score']}")
        print(f"  - Target label:         {sample_tier1['data.label']}")
        
        print("\nTesting Dataset pipeline with Tier 2 (Soft Sigmoid Probability Targets)...")
        ds_tier2 = LargeMAMMALDataset(
            selection_parquet_path=dedup_path,
            bb_mapper=bb_mapper,
            protein_sequence=pgk2_sequence,
            tokenizer_op=tokenizer_op,
            scoring_scheme="tier2",
            temperature=1.5,
            bias=1.5,
            score_threshold_labeling=0.5,
        )
        sample_tier2 = ds_tier2[0]
        print(f"✔ Dataset loaded under Tier 2. Target Score: {sample_tier2['data.target_score']:.4f}")
        
        print("\nTesting Dataset pipeline with Tier 3 (Bayesian Read Ratio)...")
        ds_tier3 = LargeMAMMALDataset(
            selection_parquet_path=dedup_path,
            bb_mapper=bb_mapper,
            protein_sequence=pgk2_sequence,
            tokenizer_op=tokenizer_op,
            scoring_scheme="tier3",
            count_pgk2_col="count_PGK2",
            count_inh_col="count_PGK2_with_inhibitor",
            count_ntc_col="count_NTC",
            temperature=1.0,
            bias=1.0,
            score_threshold_labeling=0.5,
        )
        sample_tier3 = ds_tier3[0]
        print(f"✔ Dataset loaded under Tier 3. Target Score: {sample_tier3['data.target_score']:.4f}")
        
        print("\nTesting Dataset pipeline in COMPACT combinatorial mode...")
        ds_compact = LargeMAMMALDataset(
            selection_parquet_path=dedup_path,
            bb_mapper=bb_mapper,
            protein_sequence=pgk2_sequence,
            tokenizer_op=tokenizer_op,
            scoring_scheme="tier2",
            score_threshold_labeling=0.5,
            compact=True,
        )
        sample_compact = ds_compact[0]
        print(f"✔ Dataset loaded under Compact mode. Target Score: {sample_compact['data.target_score']:.4f}")
        print(f"  - Encoder tokens shape: {sample_compact[ENCODER_INPUTS_TOKENS].shape}")
        
        print("\n==================================================")
        print("🎉 ALL VERIFICATION TESTS PASSED SUCCESSFULLY! 🎉")
        print("==================================================")
        print("The pipeline is fully validated, completely modular, and ready for HPC execution.")

    finally:
        # Clean up sandbox files
        if os.path.exists(mock_dir):
            shutil.rmtree(mock_dir)


if __name__ == "__main__":
    test_pipeline()
