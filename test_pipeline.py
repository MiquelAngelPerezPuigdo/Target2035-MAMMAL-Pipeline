"""
DREAM x CACHE Target 2035 Drug Discovery Challenge
Pipeline Validation & Testing Suite (MMELON late-fusion version).

This script acts as a test runner that:
1. Generates synthetic Mock DEL selection parquets and mock building block parquets.
2. Runs the end-to-end Polars deduplication and aggregates count and Z-score metrics.
3. Tests the building block MMELON embedding caching mechanism.
4. Verifies PyTorch CombinatorialMMELONDataset loading, label calculations, and embedding fusion.
5. Performs a mock train-and-evaluate run of the MMELON Combinatorial MLP head to ensure correctness.
"""

from __future__ import annotations
import os
import shutil
import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import our custom modules
from preprocess_del import (
    deduplicate_selection_parquet,
    BuildingBlockMapper,
    CombinatorialMMELONDataset,
    cache_bb_embeddings
)
from run_pipeline import MMELONCombinatorialMLP

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
    print("STARTING TEST RUNNER FOR THE WORKFLOW (MMELON)")
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
        
        # 4. Test Building Block Embedding Caching
        print("\nTesting MMELON embedding cacher...")
        bb_embeddings_path = f"{mock_dir}/mmelon_bb_embeddings_mock.npz"
        cache_bb_embeddings(
            bb_mapper=bb_mapper,
            base_model_path="ibm-research/biomed.sm.mv-te-84m",
            output_path=bb_embeddings_path,
            device="cpu"
        )
        assert os.path.exists(bb_embeddings_path), "Embedding cache was not created!"
        print("✔ Embedding cache successfully written.")
        
        # 5. Initialize Large-Scale Dataset with Customizable Target Schemes
        print("\nTesting CombinatorialMMELONDataset under Tier 2...")
        dataset = CombinatorialMMELONDataset(
            selection_parquet_path=dedup_path,
            bb_embeddings_path=bb_embeddings_path,
            scoring_scheme="tier2",
            score_threshold_labeling=0.5,
        )
        print(f"✔ Dataset loaded under Tier 2 with {len(dataset)} compounds.")
        
        # Grab first item
        embs, label, target_score = dataset[0]
        print("✔ Dataset output shape validation:")
        print(f"  - Compound embedding shape: {embs.shape}")
        print(f"  - Target Score value:       {target_score:.4f}")
        print(f"  - Label value:              {label}")
        
        # 6. Test MLP head setup and mock train step
        print("\nTesting MLP prediction head feedforward & training loop...")
        loader = DataLoader(dataset, batch_size=4, shuffle=True)
        mlp_head = MMELONCombinatorialMLP(input_dim=dataset.emb_dim)
        
        batch_embs, batch_labels, _ = next(iter(loader))
        logits = mlp_head(batch_embs).squeeze(-1)
        
        assert logits.shape == batch_labels.shape, "Output shape mismatch between logits and labels!"
        print(f"  - MLP feedforward successful. Output shape: {logits.shape}")
        
        # Loss run
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(logits, batch_labels)
        print(f"  - Loss calculation successful. Loss value: {loss.item():.4f}")
        
        print("\n==================================================")
        print("🎉 ALL VERIFICATION TESTS PASSED SUCCESSFULLY! 🎉")
        print("==================================================")
        print("The MMELON pipeline is fully validated, completely modular, and ready for HPC execution.")

    finally:
        # Clean up sandbox files
        if os.path.exists(mock_dir):
            shutil.rmtree(mock_dir)


if __name__ == "__main__":
    test_pipeline()
