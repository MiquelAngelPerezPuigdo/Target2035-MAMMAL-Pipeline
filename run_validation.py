#!/usr/bin/env python3
"""
DREAM x CACHE Target 2035 Drug Discovery Challenge
Unified Model Inference & Validation/Submission Pipeline.

This script allows your collaborator to load a fine-tuned MAMMAL model,
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
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from fuse.data.tokenizers.modular_tokenizer.op import ModularTokenizerOp
from fuse.data.utils.collates import CollateDefault
from mammal.model import Mammal
from mammal.keys import ENCODER_INPUTS_STR, ENCODER_INPUTS_TOKENS, ENCODER_INPUTS_ATTENTION_MASK, SCORES

# Import official evaluation helper if available
sys.path.append(str(Path(__file__).parent / "Target2035_Aircheck_Utils" / "EvaluationCode"))
try:
    from evaluation_function import evaluate_team_model
    EVALUATION_AVAILABLE = True
except ImportError:
    EVALUATION_AVAILABLE = False

from preprocess_del import load_pgk2_sequence, construct_mammal_prompt, BuildingBlockMapper

# ---------------------------------------------------------------------------
# 1. Dataset for Validation / Inference (No Training Labels Required)
# ---------------------------------------------------------------------------

class InferenceDataset(Dataset):
    """
    Dataset representing the validation or test compounds for MAMMAL inference.
    Maps molecular structures alongside CatalogIDs and target protein sequences.
    """
    def __init__(
        self,
        filepath: str,
        protein_sequence: str,
        tokenizer_op: ModularTokenizerOp,
        bb_mapper: BuildingBlockMapper | None = None,
        compact: bool = False,
        id_column: str = "CatalogID",
        smiles_column: str = "SMILES",
    ) -> None:
        self.tokenizer_op = tokenizer_op
        self.protein_sequence = protein_sequence
        self.bb_mapper = bb_mapper
        self.compact = compact
        
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
        if self.compact and self.bb_mapper is not None:
            print("✔ Compact combinatorial mode active. Validation SMILES will be reverse-engineered to library building blocks.")

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, idx: int) -> dict:
        compound_id = self.ids[idx]
        compound_smiles = self.smiles[idx]
        
        # If compact/combinatorial mode and a bb_mapper is provided, we reverse engineer the building blocks!
        if self.compact and self.bb_mapper is not None:
            # Reverse engineer the SMILES into building blocks
            bb_smiles, bb_codes = self.bb_mapper.reverse_engineer_smiles(compound_smiles)
        else:
            bb_smiles = []
            bb_codes = None
            
        prompt_str = construct_mammal_prompt(
            protein_sequence=self.protein_sequence,
            bb_smiles=bb_smiles,
            compound_smiles=compound_smiles,
            bb_codes=bb_codes,
            compact=self.compact
        )
        
        sample_dict = {
            "data.sample_id": compound_id,
            "data.smiles": compound_smiles,
            ENCODER_INPUTS_STR: prompt_str
        }
        
        self.tokenizer_op(
            sample_dict=sample_dict,
            key_in=ENCODER_INPUTS_STR,
            key_out_tokens_ids=ENCODER_INPUTS_TOKENS,
            key_out_attention_mask=ENCODER_INPUTS_ATTENTION_MASK,
            max_seq_len=512,
        )
        
        for k in (ENCODER_INPUTS_TOKENS, ENCODER_INPUTS_ATTENTION_MASK):
            sample_dict[k] = torch.tensor(sample_dict[k])
            
        return sample_dict


def run_mammal_inference(
    model: Mammal,
    dataloader: DataLoader,
    tokenizer_op: ModularTokenizerOp,
    device: str = "cpu",
    classification_position: int = 1,
) -> pd.DataFrame:
    """Run model feedforward and extract the probabilities of the active class token '<1>'."""
    model.eval()
    model = model.to(device)

    neg_token_id = tokenizer_op.get_token_id("<0>")
    pos_token_id = tokenizer_op.get_token_id("<1>")

    results = {
        "CatalogID": [],
        "SMILES": [],
        "Score": []
    }
    
    print(f"Running inference on device: {device}...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting Binders"):
            batch_size = batch[ENCODER_INPUTS_TOKENS].shape[0]

            for k in (ENCODER_INPUTS_TOKENS, ENCODER_INPUTS_ATTENTION_MASK):
                batch[k] = batch[k].to(device)
                
            batch_out = model.generate(
                batch,
                output_scores=True,
                return_dict_in_generate=True,
                max_new_tokens=5,
            )

            decoder_scores = batch_out[SCORES]  # Dimensions: (B, seq_len, vocab_size)
            
            for i in range(batch_size):
                compound_id = batch["data.sample_id"][i]
                smiles = batch["data.smiles"][i]
                
                # Fetch raw prediction logits at the target token position
                decoder_logits = decoder_scores[i].cpu().numpy()  # (seq_len, vocab_size)
                
                neg_logit = decoder_logits[classification_position, neg_token_id]
                pos_logit = decoder_logits[classification_position, pos_token_id]
                
                # Softmax normalization over the binary decision labels
                score = pos_logit / (pos_logit + neg_logit + 1e-8)
                
                results["CatalogID"].append(compound_id)
                results["SMILES"].append(smiles)
                results["Score"].append(score)

    return pd.DataFrame(results)

# ---------------------------------------------------------------------------
# 2. Main CLI Controller
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DREAM x CACHE Target 2035: Model Inference, Evaluation, & Submission Pipeline CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model and Inputs
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Path to the directory containing the fine-tuned MAMMAL checkpoint and saved tokenizer."
    )
    parser.add_argument(
        "--validation-file",
        type=str,
        required=True,
        help="Path to the validation or test split CSV/Parquet (e.g. PGK2_CACHE_Val_Test_Set.csv)."
    )
    parser.add_argument(
        "--fasta-file",
        type=str,
        default="pgk2_sequence.fasta",
        help="Path to the PGK2 FASTA protein sequence."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="submissions",
        help="Directory where submission outputs (.txt and .csv files) will be saved."
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Use a compact combinatorial prompt style by reverse-engineering validation SMILES to library building blocks."
    )
    parser.add_argument(
        "--bb-glob",
        type=str,
        default="OpenDEL-libraries/building_blocks/*.parquet",
        help="Glob pattern pointing to the building block parquet files (required if --compact is active)."
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
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader num workers."
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
    print("   DREAM x CACHE TARGET 2035 - VALIDATION & INFERENCE CLI")
    print("=" * 60)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Hardware accelerator: {device.upper()}")
    
    # 1. Load fine-tuned model and saved tokenizer from model_dir
    print(f"\nLoading fine-tuned model from '{args.model_dir}'...")
    try:
        model = Mammal.from_pretrained(args.model_dir)
        tokenizer_op = ModularTokenizerOp.from_pretrained(args.model_dir)
        print("✔ Fine-tuned model and tokenizer successfully loaded.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Please ensure --model-dir points to a valid checkpoint containing config.json and tokenizer directory.")
        return
        
    # 2. Get PGK2 Sequence Context
    pgk2_sequence = load_pgk2_sequence(args.fasta_file)
    print(f"✔ Loaded PGK2 sequence context.")
    
    # Load BuildingBlockMapper if compact mode is selected
    bb_mapper = None
    if args.compact:
        print(f"\nLoading building blocks for reverse engineering from '{args.bb_glob}'...")
        bb_mapper = BuildingBlockMapper(bb_files_glob=args.bb_glob)
        if len(bb_mapper.bb_map) == 0:
            print("⚠️ Warning: No building blocks mapped for reverse-engineering. Ensure --bb-glob is correct.")
        else:
            print(f"✔ Successfully loaded {len(bb_mapper.bb_map):,} building blocks for reverse engineering.")
            
    # 3. Create Dataset and DataLoader
    try:
        dataset = InferenceDataset(
            filepath=args.validation_file,
            protein_sequence=pgk2_sequence,
            tokenizer_op=tokenizer_op,
            bb_mapper=bb_mapper,
            compact=args.compact,
            id_column=args.id_col,
            smiles_column=args.smiles_col
        )
        
        # Setup padding crop optimization
        pad_id = tokenizer_op.get_token_id("<PAD>")
        special_handlers = {
            ENCODER_INPUTS_TOKENS: partial(CollateDefault.crop_padding, pad_token_id=pad_id),
            ENCODER_INPUTS_ATTENTION_MASK: partial(CollateDefault.crop_padding, pad_token_id=False),
        }
        
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            collate_fn=CollateDefault(special_handlers_keys=special_handlers),
            shuffle=False,
            num_workers=args.num_workers,
        )
    except Exception as e:
        print(f"❌ Error preparing dataloader: {e}")
        return
        
    # 4. Run Model Prediction
    print("\nRunning inference engine...")
    pred_df = run_mammal_inference(
        model=model,
        dataloader=dataloader,
        tokenizer_op=tokenizer_op,
        device=device
    )
    
    # 5. Format & Save Challenge Submission Outputs
    print("\nFormatting submission files...")
    
    # Sort descending by prediction score
    ranked_df = pred_df.sort_values(by="Score", ascending=False).reset_index(drop=True)
    
    # Identify top 50 candidate binders (flag Sel_50 as 1, others as 0)
    ranked_df["Sel_50"] = 0
    ranked_df.loc[:49, "Sel_50"] = 1
    
    # Output file paths
    val_txt_path = os.path.join(args.output_dir, "Team_MAMMAL_submission_validation.txt")
    test_csv_path = os.path.join(args.output_dir, "Team_MAMMAL_submission_test.csv")
    
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
    
    # 6. Run Local Performance Evaluation (If labels are available)
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
