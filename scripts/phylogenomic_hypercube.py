#!/usr/bin/env python3
"""
3D Phylogenomic Matrix: Gene A Divergence across 100 Species

Generates an interactive 3D scatter plot visualizing a phylogenomic hypercube
for Gene A divergence across 100 species. Prioritizes sequence divergence as
the key metric.

Model:
  X-axis: 100 species sorted by phylogenetic distance
  Y-axis: Locus index (0 = ancient ortholog, 1+ = paralogs ranked by divergence)
  Z-axis: Sequence divergence score (Ks/dS-like, 0.0-0.8)
  Color:  Divergence (Viridis)
  Symbol: Ancient vs Paralogues
  Size:   Confidence (1.0 - divergence * 0.5)

Usage:
  python scripts/phylogenomic_hypercube.py
"""

# ── Imports ──────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import plotly.express as px

# ── Configuration ────────────────────────────────────────────────────────────
N_SPECIES    = 100    # Number of independent species
MIN_PARALOGS = 1      # Minimum paralogs per species
MAX_PARALOGS = 5      # Maximum paralogs per species
DIV_LOW      = 0.1    # Minimum paralog divergence (simulating Ks or dS)
DIV_HIGH     = 0.8    # Maximum paralog divergence
SEED         = 42     # Random seed for reproducibility

# ── Synthetic data generation ────────────────────────────────────────────────
np.random.seed(SEED)

rows = []

for sp_idx in range(N_SPECIES):
    species_name = f"Species_{sp_idx:03d}"

    # Ancient locus: the reference ortholog with zero divergence
    rows.append({
        "Species":     species_name,
        "Locus_Type":  "Ancient",
        "Locus_Index": 0,
        "Divergence":  0.0,
        "Confidence":  1.0,
    })

    # Generate 1-5 random paralogs per species
    n_paralogs = np.random.randint(MIN_PARALOGS, MAX_PARALOGS + 1)

    # Paralogs have divergence in [0.1, 0.8], shaped by phylogenetic distance
    phylo_factor = (sp_idx + 1) / N_SPECIES
    alpha_param = 2.0
    beta_param  = max(1.0, 5.0 - 4.0 * phylo_factor)
    raw_divs = np.random.beta(alpha_param, beta_param, size=n_paralogs)
    paralog_divs = DIV_LOW + raw_divs * (DIV_HIGH - DIV_LOW)

    # Sort by divergence (rank paralogs)
    paralog_divs = np.sort(paralog_divs)

    for p_idx, div in enumerate(paralog_divs, start=1):
        # Confidence decreases with divergence
        confidence = 1.0 - div * 0.5
        rows.append({
            "Species":     species_name,
            "Locus_Type":  "Paralogues",
            "Locus_Index": p_idx,
            "Divergence":  round(float(div), 4),
            "Confidence":  round(float(confidence), 4),
        })

# ── Build tidy DataFrame ────────────────────────────────────────────────────
df = pd.DataFrame(rows)

print(f"DataFrame: {len(df)} rows x {len(df.columns)} columns")
print(f"  Ancient loci: {(df.Locus_Type == 'Ancient').sum()}")
print(f"  Paralogs:     {(df.Locus_Type == 'Paralogues').sum()}")

# ── 3D Scatter Plot ─────────────────────────────────────────────────────────
fig = px.scatter_3d(
    df,
    x="Species",
    y="Locus_Index",
    z="Divergence",
    color="Divergence",
    symbol="Locus_Type",
    size="Confidence",
    color_continuous_scale="Viridis",
    title="3D Phylogenomic Matrix: Gene A Divergence across 100 Species",
    labels={
        "Species":     "Species",
        "Locus_Index": "Locus (0=Ancient)",
        "Divergence":  "Sequence Distance",
    },
    hover_data=["Species", "Locus_Type", "Locus_Index", "Divergence", "Confidence"],
)

# ── Optimize layout for 100 species ─────────────────────────────────────────
fig.update_layout(
    scene=dict(
        xaxis=dict(
            title="Species",
            nticks=10,
            showticklabels=False,
        ),
        yaxis=dict(title="Locus (0=Ancient)"),
        zaxis=dict(title="Sequence Distance"),
    ),
    width=1000,
    height=750,
    margin=dict(l=0, r=0, b=0, t=40),
)

# ── Show interactive plot ────────────────────────────────────────────────────
fig.show()
