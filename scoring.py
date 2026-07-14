"""
DREAM x CACHE Target 2035 Drug Discovery Challenge
Scoring Functions Mapping and Target Engineering Module.

This module contains flexible, customizable formulations to map DNA-Encoded Library (DEL)
selection experimental metrics under multiple conditions (Target, Inhibitor, No-Target Control)
into a unified scoring representation (0 to 1) for MMELON model training.

NOTE: This scoring function script is the PRIMARY arena for experimentation and target engineering. 
Adjusting thresholds, parameters, and formula weights here is where the "play" needs to be done 
to find the optimal signal-to-noise ratio for training.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def sigmoid(x: np.ndarray | pd.Series, temperature: float = 1.0, bias: float = 0.0) -> np.ndarray:
    """Standard parameterized Sigmoid activation function."""
    return 1.0 / (1.0 + np.exp(-temperature * (x - bias)))


def score_tier2_soft_sigmoid(
    df: pd.DataFrame,
    z_pgk2_col: str = "zscore_PGK2",
    z_inh_col: str = "zscore_PGK2_with_inhibitor",
    z_ntc_col: str = "zscore_NTC",
    temperature: float = 1.5,
    bias: float = 1.5,
) -> np.ndarray:
    """
    Tier 2: Specificity Difference Score (Continuous Soft-Labeling) - STRONGLY RECOMMENDED.
    
    Computes a raw competitive difference score:
      S = Z_PGK2 - max(Z_NTC, Z_inh)
    And maps it to a continuous 0-1 range using a parameterized Sigmoid.
    
    TIP: A target score threshold of 0.5 (under standard temperature=1.5 and bias=1.5)
    provides an mathematically optimal midpoint boundary to separate pocket-specific 
    binders from non-specific ones.
    """
    z_pgk2 = df[z_pgk2_col] if z_pgk2_col in df.columns else 0.0
    z_inh = df[z_inh_col] if z_inh_col in df.columns else 0.0
    z_ntc = df[z_ntc_col] if z_ntc_col in df.columns else 0.0

    raw_diff = z_pgk2 - np.maximum(z_ntc, z_inh)
    return sigmoid(raw_diff, temperature=temperature, bias=bias).to_numpy()


def map_scores(
    df: pd.DataFrame,
    scheme: str = "tier2",
    **kwargs,
) -> np.ndarray:
    """
    Modular entry point to map raw experimental values to a 0-1 target value.
    
    This is where model performance is decided. Experimenting with different sigmoid 
    temperatures/biases or adding weights to controls is highly encouraged!
    """
    scheme_clean = scheme.lower().replace("_", "")
    if scheme_clean in ("tier2", "softsigmoid"):
        return score_tier2_soft_sigmoid(df, **kwargs)
    else:
        raise ValueError(f"Unknown scoring scheme: '{scheme}'. 'tier2' (soft_sigmoid) is the primary active scheme.")

