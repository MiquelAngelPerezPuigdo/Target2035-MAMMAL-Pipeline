"""
DREAM x CACHE Target 2035 Drug Discovery Challenge
DEL Data Housekeeping, Deduplication, and MAMMAL Prompt Preparation Pipeline.

This script provides robust utility functions and an end-to-end pipeline to:
1. Deduplicate the raw DEL selection data using Polars (combining z-scores with Stouffer's method).
2. Compute custom target labels/scores reflecting binding specificity.
3. Parse compound IDs to look up building block structures.
4. Construct sequence-to-sequence prompt encodings suitable for multi-modal MAMMAL fine-tuning.
"""

from __future__ import annotations
import os
import glob
import warnings
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import polars as pl
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import scoring

# Default regex matching the BCM compound ID pattern
BCM_COMPOUND_PATTERN = r"^[a-zA-Z]+\d+(?:_\d+)?(?:-\d+){3,}(?:-\d+)?$"

# ---------------------------------------------------------------------------
# 1. High-Performance Deduplication (Polars-Powered)
# ---------------------------------------------------------------------------

_POLARS_AGG = {
    "sum": lambda col: pl.col(col).sum(),
    "mean": lambda col: pl.col(col).mean(),
    "max": lambda col: pl.col(col).max(),
    "min": lambda col: pl.col(col).min(),
    "stouffer": lambda col: (
        pl.col(col).sum() / pl.col(col).count().cast(pl.Float64).sqrt()
    ).alias(col),
}

def combine_zscores_stouffer(z_scores: np.ndarray) -> float:
    """Combine independent Z-scores using unweighted Stouffer's method."""
    z_scores = np.asarray(z_scores, dtype=float)
    if z_scores.size == 0:
        return float("nan")
    if z_scores.size == 1:
        return float(z_scores[0])
    return float(z_scores.sum() / np.sqrt(z_scores.size))

def deduplicate_selection_parquet(
    input_path: str,
    output_path: str,
    dedup_col: str = "SMILES",
    compound_col: str = "compound",
) -> None:
    """
    Load a raw selection parquet, clean invalid structures, deduplicate by SMILES,
    and aggregate counts (sum) and Z-scores (Stouffer's method).
    """
    print(f"Loading raw selection file: {input_path}")
    df = pl.read_parquet(input_path)
    original_rows = len(df)
    
    # 1. Cleaning: drop nulls and validate SMILES/compounds
    print("Cleaning records...")
    df = df.filter(pl.col(dedup_col).is_not_null())
    df = df.filter(
        pl.col(dedup_col).str.contains("[Cc]") & 
        (pl.col(dedup_col).str.len_chars() > 10)
    )
    if compound_col in df.columns:
        df = df.filter(pl.col(compound_col).str.contains(BCM_COMPOUND_PATTERN))
    
    clean_rows = len(df)
    
    # 2. Map columns to aggregation schemes
    agg_scheme = {}
    for col in df.columns:
        if col == dedup_col:
            continue
        if col.endswith("_count"):
            agg_scheme[col] = "sum"
        elif col.endswith("_zscore") or col.endswith("_score"):
            agg_scheme[col] = "stouffer"
        elif col == "historic_hits":
            agg_scheme[col] = "max"

    print(f"Deduplicating by {dedup_col} and aggregating with rules: {agg_scheme}")
    
    exprs = [_POLARS_AGG[method](col) for col, method in agg_scheme.items()]
    # Keep the first seen value for columns that are not aggregated or the group key
    for col in df.columns:
        if col != dedup_col and col not in agg_scheme:
            exprs.append(pl.col(col).first())
            
    dedup_df = df.group_by(dedup_col).agg(exprs)
    
    # Keep input column order
    col_order = [c for c in df.columns if c in dedup_df.columns]
    dedup_df = dedup_df.select(col_order)
    
    print(f"Writing deduplicated output to: {output_path}")
    dedup_df.write_parquet(output_path, compression="zstd")
    
    print("\n── Deduplication Metrics ──")
    print(f"Original rows   : {original_rows:,}")
    print(f"Cleaned rows    : {clean_rows:,} ({ (clean_rows - original_rows)/original_rows*100:+.2f}%)")
    print(f"Deduplicated    : {len(dedup_df):,} ({ (len(dedup_df) - original_rows)/original_rows*100:+.2f}%)")
    print("───────────────────────────")

# ---------------------------------------------------------------------------
# 2. Target Label & Binding Specificity Engineering
# ---------------------------------------------------------------------------

def compute_binding_scores(
    df: pd.DataFrame,
    z_pgk2_col: str = "zscore_PGK2",
    z_inh_col: str = "zscore_PGK2_with_inhibitor",
    z_ntc_col: str = "zscore_NTC",
) -> pd.DataFrame:
    """
    Compute specific binding metrics mapping 3 experimental conditions into train targets.
    
    - Target Affinity Score (TAS) = Z_PGK2 - Z_NTC
    - ATP Specificity Score = Z_PGK2 - Z_inh
    - Combined Competitive Score = Z_PGK2 - max(Z_NTC, Z_inh)
    """
    df = df.copy()
    
    # Handle optional missing columns gracefully
    z_pgk2 = df[z_pgk2_col] if z_pgk2_col in df.columns else 0.0
    z_inh = df[z_inh_col] if z_inh_col in df.columns else 0.0
    z_ntc = df[z_ntc_col] if z_ntc_col in df.columns else 0.0
    
    df["target_affinity_score"] = z_pgk2 - z_ntc
    df["atp_specificity_score"] = z_pgk2 - z_inh
    df["combined_competitive_score"] = z_pgk2 - np.maximum(z_ntc, z_inh)
    
    return df

def generate_target_scores(
    df: pd.DataFrame,
    scheme: str = "tier2",
    **kwargs,
) -> pd.DataFrame:
    """
    Map raw selection experimental conditions to a 0-1 continuous or binary target label.
    Delegates calculation to the custom scoring engine module.
    """
    df = df.copy()
    df["target_score"] = scoring.map_scores(df, scheme=scheme, **kwargs)
    return df

def load_pgk2_sequence(fasta_path: str | None = None) -> str:
    """Load the Human Phosphoglycerate kinase 2 (PGK2) amino acid sequence."""
    if fasta_path is None:
        fasta_path = os.path.join(os.path.dirname(__file__), "pgk2_sequence.fasta")
    if not os.path.exists(fasta_path):
        return "MSLSKKLTLDKLDVRGKRVIMRVDFNVPMKKNQITNNQRIKASIPSIKYCLDNGAKAVVLMSHLGRPDGVPMPDKYSLAPVAVELKSLLGKDVLFLKDCVGAEVEKACANPAPGSVILLENLRFHVEEEGKGQDPSGKKIKAEPDKIEAFRASLSKLGDVYVNDAFGTAHRAHSSMVGVNLPHKASGFLMKKELDYFAKALENPVRPFLAILGGAKVADKIQLIKNMLDKVNEMIIGGGMAYTFLKVLNNMEIGASLFDEEGAKIVKDIMAKAQKNGVRITFPVDFVTGDKFDENAQVGKATVASGISPGWMGLDCGPESNKNHAQVVAQARLIVWNGPLGVFEWDAFAKGTKALMDEIVKATSKGCITVIGGGDTATCCAKWNTEDKVSHVSTGGGASLELLEGKILPGVEALSNM"
    seq_lines = []
    with open(fasta_path, "r") as f:
        for line in f:
            if line.startswith(">"):
                continue
            seq_lines.append(line.strip())
    return "".join(seq_lines)

def generate_binary_labels(
    df: pd.DataFrame,
    score_col: str = "combined_competitive_score",
    threshold: float = 2.0,
) -> pd.DataFrame:
    """Categorize compounds into active (1) or inactive (0) based on score threshold."""
    df = df.copy()
    df["label"] = (df[score_col] >= threshold).astype(int)
    return df

# ---------------------------------------------------------------------------
# 3. Combinatorial Building Block Mapping
# ---------------------------------------------------------------------------

class BuildingBlockMapper:
    """Parses Compound IDs and maps them to physical building block SMILES."""
    
    def __init__(self, bb_files_glob: str) -> None:
        """
        Initialize mapper by loading building block composition files.
        e.g., 'OpenDEL-libraries/building_blocks/*.parquet'
        """
        self.bb_map = {}
        self.bb_cycle_maps = {1: {}, 2: {}, 3: {}}
        bb_files = glob.glob(bb_files_glob)
        if not bb_files:
            warnings.warn(f"No building block files found matching {bb_files_glob}")
            return
            
        print(f"Loading {len(bb_files)} building block lookup tables...")
        for file in bb_files:
            try:
                # Expect columns: 'ID' (or 'bb_id') and 'SMILES' (or 'smiles')
                bb_df = pl.read_parquet(file)
                id_col = [c for c in bb_df.columns if c.lower() in ("id", "bb_id", "bb_name")][0]
                smiles_col = [c for c in bb_df.columns if c.lower() in ("smiles", "structure")][0]
                
                # Determine cycle from filename/path
                filename = os.path.basename(file).lower()
                cycle = None
                if "bb1" in filename or "cycle1" in filename:
                    cycle = 1
                elif "bb2" in filename or "cycle2" in filename:
                    cycle = 2
                elif "bb3" in filename or "cycle3" in filename:
                    cycle = 3
                
                for row in bb_df.select([id_col, smiles_col]).iter_rows():
                    bb_id_str = str(row[0])
                    smiles_str = str(row[1])
                    self.bb_map[bb_id_str] = smiles_str
                    if cycle is not None:
                        self.bb_cycle_maps[cycle][bb_id_str] = smiles_str
            except Exception as e:
                print(f"Error loading {file}: {e}")
                
        print(f"Loaded {len(self.bb_map):,} unique building block mappings.")
        for cycle_idx, cycle_map in self.bb_cycle_maps.items():
            print(f"  Cycle {cycle_idx}: {len(cycle_map):,} building blocks mapped.")

    def get_bb_smiles(self, compound_id: str) -> list[str]:
        """
        Split compound ID (e.g., 'qDOS11-0012-0045-0210') and return smiles of BBs.
        If a building block structure is missing, returns None for that position.
        """
        try:
            parts = compound_id.split("-")
            # Usually format is: Library_ID, BB1_ID, BB2_ID, BB3_ID
            bb_ids = parts[1:4]
            return [self.bb_map.get(bb_id, "") for bb_id in bb_ids]
        except Exception:
            return ["", "", ""]

    def _init_reverse_engineering(self) -> None:
        """Initialize and cache fingerprints for building blocks to speed up reverse engineering."""
        if hasattr(self, "_re_initialized") and self._re_initialized:
            return
            
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
        except ImportError:
            raise ImportError(
                "RDKit is required for building block reverse-engineering. "
                "Please run `pip install rdkit` to install it."
            )
            
        print("Initializing building block fingerprint cache for reverse engineering...")
        self.bb_mols = {}
        self.bb_fps = {}
        
        for bb_id, smiles in self.bb_map.items():
            if not smiles:
                continue
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    self.bb_mols[bb_id] = mol
                    self.bb_fps[bb_id] = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            except Exception:
                pass
                
        self._re_initialized = True
        print(f"Cached fingerprints for {len(self.bb_fps)} building blocks.")

    def reverse_engineer_smiles(self, query_smiles: str) -> tuple[list[str], list[str]]:
        """
        Decompose a full molecule SMILES (e.g., from validation set) into 
        the 3 closest building blocks from our library.
        
        Returns:
            bb_smiles: list of 3 SMILES strings
            bb_codes: list of 3 building block IDs
        """
        self._init_reverse_engineering()
        
        from rdkit import Chem
        from rdkit.Chem import AllChem, DataStructs
        
        query_mol = Chem.MolFromSmiles(query_smiles)
        if query_mol is None:
            return ["", "", ""], ["", "", ""]
            
        query_fp = AllChem.GetMorganFingerprintAsBitVect(query_mol, 2, nBits=2048)
        
        bb_smiles = ["", "", ""]
        bb_codes = ["", "", ""]
        
        # Check if we have cycle-specific building blocks mapped
        has_cycles = all(len(self.bb_cycle_maps[i]) > 0 for i in (1, 2, 3))
        
        if has_cycles:
            # For each cycle (1, 2, 3), find the best matching building block
            for cycle_idx in (1, 2, 3):
                candidates = self.bb_cycle_maps[cycle_idx]
                best_id = ""
                best_score = -1.0
                
                for bb_id, bb_sm in candidates.items():
                    if bb_id not in self.bb_fps:
                        continue
                        
                    bb_mol = self.bb_mols[bb_id]
                    bb_fp = self.bb_fps[bb_id]
                    
                    # 1. Substructure check (highest priority match)
                    try:
                        has_match = query_mol.HasSubstructMatch(bb_mol)
                    except Exception:
                        has_match = False
                        
                    # 2. Tversky similarity for sub-fingerprint match
                    try:
                        sim = DataStructs.TverskySimilarity(bb_fp, query_fp, 0.0, 1.0)
                    except Exception:
                        sim = 0.0
                        
                    # Combine scores: substructure matches get boosted
                    score = 1.0 + sim if has_match else sim
                    
                    if score > best_score:
                        best_score = score
                        best_id = bb_id
                        
                if best_id:
                    bb_codes[cycle_idx - 1] = best_id
                    bb_smiles[cycle_idx - 1] = candidates[best_id]
        else:
            # Fallback flat lookup: find top 3 overall matches
            all_scores = []
            for bb_id, bb_sm in self.bb_map.items():
                if bb_id not in self.bb_fps:
                    continue
                    
                bb_mol = self.bb_mols[bb_id]
                bb_fp = self.bb_fps[bb_id]
                
                try:
                    has_match = query_mol.HasSubstructMatch(bb_mol)
                except Exception:
                    has_match = False
                    
                try:
                    sim = DataStructs.TverskySimilarity(bb_fp, query_fp, 0.0, 1.0)
                except Exception:
                    sim = 0.0
                    
                score = 1.0 + sim if has_match else sim
                all_scores.append((score, bb_id, bb_sm))
                
            all_scores.sort(key=lambda x: x[0], reverse=True)
            top_3 = all_scores[:3]
            for idx, (score, bb_id, bb_sm) in enumerate(top_3):
                if idx < 3:
                    bb_codes[idx] = bb_id
                    bb_smiles[idx] = bb_sm
                    
        return bb_smiles, bb_codes

# ---------------------------------------------------------------------------
# 4. Multi-Modal MAMMAL Prompt Encoder
# ---------------------------------------------------------------------------

def construct_mammal_prompt(
    protein_sequence: str,
    bb_smiles: list[str],
    compound_smiles: str,
    bb_codes: list[str] | None = None,
    compact: bool = False,
) -> str:
    """
    Format combinatorial building blocks, final compound SMILES, and target protein
    sequence into a single unified MAMMAL sequence-to-sequence encoder prompt.
    """
    # 1. Target Protein context
    protein_part = (
        f"<@TOKENIZER-TYPE=AA><MOLECULAR_ENTITY><MOLECULAR_ENTITY_GENERAL_PROTEIN>"
        f"<SEQUENCE_NATURAL_START>{protein_sequence}<SEQUENCE_NATURAL_END>"
    )
    
    # 2. Combinatorial building blocks
    bbs_combined = ".".join([sm for sm in bb_smiles if sm])
    bb_part = ""
    if bbs_combined:
        bb_part = (
            f"<@TOKENIZER-TYPE=SMILES><MOLECULAR_ENTITY><MOLECULAR_ENTITY_SMALL_MOLECULE>"
            f"<SEQUENCE_NATURAL_START>{bbs_combined}<SEQUENCE_NATURAL_END>"
        )
        
    if compact:
        # 3. BB codes representation instead of holistic compound SMILES
        if bb_codes is None:
            bb_codes = ["", "", ""]
        bb_codes_str = "-".join([str(c) for c in bb_codes if c])
        compound_part = (
            f"<@TOKENIZER-TYPE=SMILES><MOLECULAR_ENTITY><MOLECULAR_ENTITY_SMALL_MOLECULE>"
            f"<SEQUENCE_NATURAL_START>{bb_codes_str}<SEQUENCE_NATURAL_END>"
        )
    else:
        # 3. Holistic Compound SMILES
        compound_part = (
            f"<@TOKENIZER-TYPE=SMILES><MOLECULAR_ENTITY><MOLECULAR_ENTITY_SMALL_MOLECULE>"
            f"<SEQUENCE_NATURAL_START>{compound_smiles}<SEQUENCE_NATURAL_END>"
        )
    
    return f"{protein_part}{bb_part}{compound_part}<EOS>"

# ---------------------------------------------------------------------------
# 5. PyTorch Streaming Dataset for Scale
# ---------------------------------------------------------------------------

class LargeMAMMALDataset(Dataset):
    """
    Memory-efficient PyTorch Dataset that loads deduplicated selection data,
    computes scores, performs building-block lookups, and prepares MAMMAL prompts on-the-fly.
    """
    
    def __init__(
        self,
        selection_parquet_path: str,
        bb_mapper: BuildingBlockMapper,
        protein_sequence: str,
        tokenizer_op: Any,
        scoring_scheme: str = "tier2",
        score_threshold_labeling: float = 0.5,
        sample_size: int | None = None,
        random_seed: int = 42,
        compact: bool = False,
        **scoring_kwargs,
    ) -> None:
        self.bb_mapper = bb_mapper
        self.protein_sequence = protein_sequence
        self.tokenizer_op = tokenizer_op
        self.compact = compact
        
        # Load and compute customizable targets
        df = pd.read_parquet(selection_parquet_path)
        df = generate_target_scores(df, scheme=scoring_scheme, **scoring_kwargs)
        
        # Binary assignment for classification metrics monitoring
        df["label"] = (df["target_score"] >= score_threshold_labeling).astype(int)
        
        # Optional balanced downsampling to manage scale
        if sample_size is not None:
            actives = df[df["label"] == 1]
            inactives = df[df["label"] == 0]
            
            # Keep all actives, and sample down inactives
            n_actives = len(actives)
            n_inactives_to_sample = min(len(inactives), sample_size - n_actives)
            
            if n_inactives_to_sample > 0:
                inactives_sampled = inactives.sample(n=n_inactives_to_sample, random_state=random_seed)
                df = pd.concat([actives, inactives_sampled]).sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
                
        self.data = df
        print(f"Dataset initialized: {len(self.data):,} items ({len(df[df['label'] == 1]):,} actives).")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        row = self.data.iloc[idx]
        compound_id = row.get("compound", "")
        compound_smiles = row.get("SMILES", "")
        label = row.get("label", 0)
        target_score = row.get("target_score", 0.0)
        
        # Combinatorial building-block lookup
        bb_smiles = self.bb_mapper.get_bb_smiles(compound_id)
        
        # Extract BB codes
        parts = str(compound_id).split("-")
        bb_codes = parts[1:4] if len(parts) >= 4 else ["", "", ""]
        
        # Construct unified prompt string
        prompt_str = construct_mammal_prompt(
            protein_sequence=self.protein_sequence,
            bb_smiles=bb_smiles,
            compound_smiles=compound_smiles,
            bb_codes=bb_codes,
            compact=self.compact
        )
        
        # Prepare MAMMAL tokenized format dict
        sample_dict = {
            "data.sample_id": idx,
            "data.label": label,
            "data.target_score": target_score,
        }
        
        from mammal.keys import ENCODER_INPUTS_STR, ENCODER_INPUTS_TOKENS, ENCODER_INPUTS_ATTENTION_MASK, LABELS_STR, LABELS_TOKENS, LABELS_ATTENTION_MASK, DECODER_INPUTS_STR, DECODER_INPUTS_TOKENS, DECODER_INPUTS_ATTENTION_MASK
        
        sample_dict[ENCODER_INPUTS_STR] = prompt_str
        
        # Tokenize prompt string
        self.tokenizer_op(
            sample_dict=sample_dict,
            key_in=ENCODER_INPUTS_STR,
            key_out_tokens_ids=ENCODER_INPUTS_TOKENS,
            key_out_attention_mask=ENCODER_INPUTS_ATTENTION_MASK,
            max_seq_len=512,  # Multi-modal prompts might require larger seq len
        )
        
        for k in (ENCODER_INPUTS_TOKENS, ENCODER_INPUTS_ATTENTION_MASK):
            sample_dict[k] = torch.tensor(sample_dict[k])
            
        # Format labels (for fine-tuning text decoder generation)
        # Here we encode whether it's active as a text class token
        sample_dict[LABELS_STR] = f"<@TOKENIZER-TYPE=SMILES><SENTINEL_ID_0><{label}><EOS>"
        self.tokenizer_op(
            sample_dict=sample_dict,
            key_in=LABELS_STR,
            key_out_tokens_ids=LABELS_TOKENS,
            key_out_attention_mask=LABELS_ATTENTION_MASK,
            max_seq_len=4,
        )
        
        pad_id = self.tokenizer_op.get_token_id("<PAD>")
        ignore_token_value = -100
        for k in (LABELS_TOKENS, LABELS_ATTENTION_MASK):
            sample_dict[k] = torch.tensor(sample_dict[k])
            
        sample_dict[LABELS_TOKENS][
            (sample_dict[LABELS_TOKENS][..., None] == torch.tensor(pad_id))
            .any(-1)
            .nonzero()
        ] = ignore_token_value

        # Format decoder inputs
        sample_dict[DECODER_INPUTS_STR] = f"<@TOKENIZER-TYPE=SMILES><DECODER_START><SENTINEL_ID_0><{label}><EOS>"
        self.tokenizer_op(
            sample_dict=sample_dict,
            key_in=DECODER_INPUTS_STR,
            key_out_tokens_ids=DECODER_INPUTS_TOKENS,
            key_out_attention_mask=DECODER_INPUTS_ATTENTION_MASK,
            max_seq_len=4,
        )
        
        for k in (DECODER_INPUTS_TOKENS, DECODER_INPUTS_ATTENTION_MASK):
            sample_dict[k] = torch.tensor(sample_dict[k])
            
        return sample_dict
