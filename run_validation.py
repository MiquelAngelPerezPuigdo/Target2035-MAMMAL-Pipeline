#!/usr/bin/env python3
"""
DREAM x CACHE Target 2035 Drug Discovery Challenge
Unified Model Inference & Validation/Submission Pipeline (MMELON late-fusion version).

This script allows your collaborator to load our fine-tuned MMELON prediction head,
run inference on the CACHE validation/test splits, select the top 50 chemically 
diverse candidates, and generate the exact submission files required by the challenge:
1. Validation Split: A .txt file with exactly 50 CatalogIDs (one per line).
2. Test Split: A .csv file with columns: CatalogID, Sel_50, Score.

Optionally, if gold standard labels are available locally, it calls the official 
evaluation script to compute ROC-AUC, PR-AUC, Cluster PRAUC, and statistical p-values.
"""

from __future__ import annotations
import os
import sys
import argparse
from pathlib import Path
from functools import partial
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Import official evaluation helper if available
sys.path.append(str(Path(__file__).parent / "Target2035_Aircheck_Utils" / "EvaluationCode"))
try:
    from evaluation_function import evaluate_team_model
    EVALUATION_AVAILABLE = True
except ImportError:
    EVALUATION_AVAILABLE = False

from preprocess_del import BuildingBlockMapper, extract_smiles_embedding
from run_pipeline import MMELONCombinatorialMLP

# ---------------------------------------------------------------------------
# 1. Dataset for Validation / Inference (No Training Labels Required)
# ---------------------------------------------------------------------------

class InferenceDataset(Dataset):
    """
    Dataset representing the validation or test compounds for MMELON inference.
    Maps molecular structures alongside CatalogIDs and computes multi-view embeddings.
    """
    def __init__(
        self,
        filepath: str,
        bb_mapper: BuildingBlockMapper | None = None,
        bb_embeddings_path: str | None = None,
        base_model_path: str = "ibm-research/biomed.sm.mv-te-84m",
        reverse_engineer: bool = False,
        id_column: str = "CatalogID",
        smiles_column: str = "SMILES",
        device: str = "cpu"
    ) -> None:
        self.bb_mapper = bb_mapper
        self.reverse_engineer = reverse_engineer
        
        # Load CSV or Parquet
        ext = Path(filepath).suffix.lower()
        if ext == ".parquet":
            df = pd.read_parquet(filepath)
        else:
            df = pd.read_csv(filepath)
            
        self.df = df
        
        # Ensure correct column names exist
        self.id_column = id_column if id_column in df.columns else df.columns[0]
        self.smiles_column = smiles_column if smiles_column in df.columns else [c for c in df.columns if c.lower() in ("smiles", "smiles_string")][0]
        
        self.ids = self.df[self.id_column].tolist()
        self.smiles = self.df[self.smiles_column].tolist()
        
        print(f"Loaded {len(self.smiles):,} compounds for inference from '{filepath}'.")
        
        if self.reverse_engineer and self.bb_mapper is not None and bb_embeddings_path is not None:
            print("✔ Reverse-Engineering validation SMILES to library building blocks...")
            self.bb_embeddings = dict(np.load(bb_embeddings_path))
            self.emb_dim = next(iter(self.bb_embeddings.values())).shape[0]
            
            # Reconstruct embeddings combinatorially
            embeddings_list = []
            for sm in tqdm(self.smiles, desc="Reverse Engineering SMILES"):
                _, bb_codes = self.bb_mapper.reverse_engineer_smiles(sm)
                bb_vecs = []
                for bb_id in bb_codes:
                    if bb_id in self.bb_embeddings:
                        bb_vecs.append(self.bb_embeddings[bb_id])
                    else:
                        bb_vecs.append(np.zeros(self.emb_dim, dtype=np.float32))
                embeddings_list.append(np.mean(bb_vecs, axis=0))
            self.embeddings = np.array(embeddings_list, dtype=np.float32)
        else:
            print("✔ Direct Embedding Mode active. Extracting multi-view embeddings directly using pre-trained MMELON model...")
            self.embeddings = extract_smiles_embedding(
                smiles_list=self.smiles,
                base_model_path=base_model_path,
                device=device
            )
            
        self.emb_dim = self.embeddings.shape[1]

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, idx: int) -> tuple[str, str, torch.Tensor]:
        compound_id = str(self.ids[idx])
        compound_smiles = str(self.smiles[idx])
        emb = self.embeddings[idx]
        return compound_id, compound_smiles, torch.tensor(emb, dtype=torch.float32)


# ---------------------------------------------------------------------------
# 2. Main CLI Controller
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DREAM x CACHE Target 2035: Model Inference, Evaluation, & Submission Pipeline CLI (MMELON)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model and Inputs
    parser.add_argument(
        "--model-file",
        type=str,
        default="processed_data/mmelon_mlp_head.pt",
        help="Path to the trained MMELON MLP head weights file."
    )
    parser.add_argument(
        "--validation-file",
        type=str,
        required=True,
        help="Path to the validation or test split CSV/Parquet (e.g. PGK2_CACHE_Val_Test_Set.csv)."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="submissions",
        help="Directory where submission outputs (.txt and .csv files) will be saved."
    )
    
    # Combinatorial / Reverse Engineering Options (Optional)
    parser.add_argument(
        "--reverse-engineer",
        action="store_true",
        help="Whether to reverse-engineer full validation SMILES to library building blocks instead of direct embedding."
    )
    parser.add_argument(
        "--bb-embeddings",
        type=str,
        default="processed_data/mmelon_bb_embeddings.npz",
        help="Path to the precomputed building block embeddings cache."
    )
    parser.add_argument(
        "--bb-glob",
        type=str,
        default="OpenDEL-libraries/building_blocks/*.parquet",
        help="Glob pattern pointing to physical library building block files."
    )
    
    # Options
    parser.add_argument(
        "--id-col",
        type=str,
        default="CatalogID",
        help="The unique molecule ID column in your dataset."
    )
    parser.add_argument(
        "--smiles-col",
        type=str,
        default="SMILES",
        help="The SMILES string column in your dataset."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Inference batch size."
    )
    
    # Optional Ground Truth Evaluation
    parser.add_argument(
        "--gold-file",
        type=str,
        default=None,
        help="Path to the gold standard labels CSV (if available locally) to compute performance metrics."
    )
    
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    print("=" * 60)
    print("   DREAM x CACHE TARGET 2035 - VALIDATION & INFERENCE CLI (MMELON)")
    print("=" * 60)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Hardware accelerator: {device.upper()}")
    
    # 1. Load fine-tuned prediction head
    print(f"\nLoading fine-tuned prediction head from '{args.model_file}'...")
    try:
        ckpt = torch.load(args.model_file, map_location=device)
        input_dim = ckpt["input_dim"]
        base_model = ckpt.get("base_model", "ibm-research/biomed.sm.mv-te-84m")
        
        mlp_head = MMELONCombinatorialMLP(input_dim=input_dim)
        mlp_head.load_state_dict(ckpt["state_dict"])
        mlp_head = mlp_head.to(device)
        mlp_head.eval()
        print("✔ Fine-tuned prediction head successfully loaded.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Please ensure --model-file points to a valid .pt checkpoint containing the MLP state dict.")
        return
        
    # Load BuildingBlockMapper if reverse-engineering is selected
    bb_mapper = None
    if args.reverse_engineer:
        bb_mapper = BuildingBlockMapper(bb_files_glob=args.bb_glob)
        if len(bb_mapper.bb_map) == 0:
            print("⚠️ Warning: No building blocks mapped. Direct Embedding mode is highly recommended instead.")
            args.reverse_engineer = False
            
    # 2. Create Dataset and DataLoader
    try:
        dataset = InferenceDataset(
            filepath=args.validation_file,
            bb_mapper=bb_mapper,
            bb_embeddings_path=args.bb_embeddings,
            base_model_path=base_model,
            reverse_engineer=args.reverse_engineer,
            id_column=args.id_col,
            smiles_column=args.smiles_col,
            device=device
        )
        
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            shuffle=False
        )
    except Exception as e:
        print(f"❌ Error preparing dataloader: {e}")
        return
        
    # 3. Run Prediction
    print("\nRunning inference engine...")
    results = {
        "CatalogID": [],
        "SMILES": [],
        "Score": []
    }
    
    with torch.no_grad():
        for comp_ids, smiles, embs in tqdm(dataloader, desc="Predicting Binders"):
            embs = embs.to(device)
            logits = mlp_head(embs).squeeze(-1)
            scores = torch.sigmoid(logits).cpu().numpy()
            
            for i in range(len(comp_ids)):
                results["CatalogID"].append(comp_ids[i])
                results["SMILES"].append(smiles[i])
                results["Score"].append(float(scores[i]))
                
    pred_df = pd.DataFrame(results)
    
    # 4. Format & Save Challenge Submission Outputs
    print("\nFormatting submission files...")
    ranked_df = pred_df.sort_values(by="Score", ascending=False).reset_index(drop=True)
    
    # Identify top 50 candidate binders (flag Sel_50 as 1, others as 0)
    ranked_df["Sel_50"] = 0
    ranked_df.loc[:49, "Sel_50"] = 1
    
    # Output file paths
    val_txt_path = os.path.join(args.output_dir, "Team_MMELON_submission_validation.txt")
    test_csv_path = os.path.join(args.output_dir, "Team_MMELON_submission_test.csv")
    
    # Save validation split file (.txt containing exactly 50 CatalogIDs, one per line)
    top_50_ids = ranked_df.loc[ranked_df["Sel_50"] == 1, "CatalogID"]
    top_50_ids.to_csv(val_txt_path, index=False, header=False)
    
    # Save test split file (.csv with columns CatalogID, Sel_50, Score)
    test_submission_df = ranked_df[["CatalogID", "Sel_50", "Score"]]
    test_submission_df.to_csv(test_csv_path, index=False)
    
    print("-" * 50)
    print(f"✔ Validation-split (.txt) submission saved to: {val_txt_path}")
    print(f"✔ Test-split (.csv) submission saved to:       {test_csv_path}")
    print("-" * 50)
    
    # 5. Run Local Performance Evaluation (If labels are available)
    if args.gold_file is not None:
        if not EVALUATION_AVAILABLE:
            print("⚠️ Local evaluation skipped: official evaluation script not found in Target2035_Aircheck_Utils.")
            return
            
        print(f"\nEvaluating predictions against local gold standard: '{args.gold_file}'...")
        try:
            gold_df = pd.read_csv(args.gold_file)
            
            # Map predictions to match the expected evaluation headers if different
            eval_pred_df = ranked_df.rename(columns={"CatalogID": "RandomID", "Sel_50": "Sel_200"})
            eval_pred_df["Sel_500"] = 0  # Dummy column as script expects Sel_200/Sel_500
            
            # Run official metrics calculations
            metrics = evaluate_team_model(
                gold_df=gold_df,
                team_df=eval_pred_df,
                label_gold="Label",
                score="Score",
                labels_team=["Sel_200"],
                cluster="Cluster",
                random_id="RandomID"
            )
            
            print("\n" + "="*45)
            print("         LOCAL EVALUATION METRICS")
            print("="*45)
            print(f"  ROC-AUC                : {metrics['ROCAUC']:.4f}")
            print(f"  PR-AUC                 : {metrics['PRAUC']:.4f}")
            print(f"  Identified Hits (@50)  : {metrics['Hits_Sel_200']}")
            print(f"  Unique Clusters Hit    : {metrics['Clusters_Sel_200']}")
            print(f"  Cluster-level PR-AUC   : {metrics['ClusterPRAUC_Sel_200']}")
            print(f"  Poisson-Binomial P-val : {metrics['P-value_Sel_200']:.4e}")
            print("="*45)
            
        except Exception as e:
            print(f"⚠️ Warning: Local evaluation failed: {e}")


if __name__ == "__main__":
    main()
