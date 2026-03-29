#!/usr/bin/env python3
"""
04 — Transcriptome Overlay (Generalized, Config-Driven)
=========================================================

QUESTION:
  Are MROH gene copies actually EXPRESSED, and if so, where? Specifically,
  are they enriched in brain regions involved in vocal learning (HVC, RA,
  Area X, LMAN)? This would connect gene copy expansion to a phenotype
  that is unique to vocal-learning species (songbirds, parrots, hummingbirds).

REASONING:
  Having hundreds of gene copies is only biologically meaningful if at least
  some copies are transcribed into mRNA and (potentially) translated into
  protein. If all copies are silenced pseudogenes, the expansion is
  evolutionarily neutral regardless of how it occurred.

  Vocal learning (the ability to imitate sounds, crucial for birdsong) is
  controlled by specialized brain nuclei:
    - HVC: Song motor pattern generator
    - RA: Robust nucleus of the arcopallium (motor output)
    - Area X: Basal ganglia song nucleus (learning/practice)
    - LMAN: Lateral magnocellular nucleus (variability/learning)

  If MROH copies are expressed in these nuclei — especially at higher levels
  than in non-song brain regions — it suggests a functional role in vocal
  learning circuits. This would make MROH the first example of a multicopy
  gene family directly linked to vocal learning.

  Data source: Colquitt et al. 2021 (Science 371:6530) published single-cell
  RNA-seq from zebra finch song-control brain regions (GEO: GSE148997).

EXPECTED FINDINGS:
  - MROH expression enriched in HVC and RA > cortex/cerebellum
  - Only a subset of copies actively transcribed (most silenced)
  - Per-copy expression heterogeneity (some copies high, most low)

  When transcriptome data is not available for the species, we generate an
  illustrative analysis showing expected patterns based on MROH6 biology.

Usage:
  python notebooks/04_transcriptome_overlay.py --species melospiza_georgiana
"""
import sys
import argparse
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent / 'scripts'))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from species_config import load_config, get_data_dirs

parser = argparse.ArgumentParser(description='Step 04: Transcriptome Overlay')
parser.add_argument('--species', required=True)
args, _ = parser.parse_known_args()

cfg = load_config(args.species)
dirs = get_data_dirs(cfg)

PROJECT = Path(cfg["_project_root"])
FIG_DIR = dirs["fig_dir"]
TABLE_DIR = dirs["table_dir"]

GENE = cfg["gene_name"]
SPECIES = cfg["common_name"]
trans_cfg = cfg.get("transcriptome", {})

sns.set_context('notebook')
sns.set_style('whitegrid')

print("=" * 70)
print(f"  STEP 04: TRANSCRIPTOME OVERLAY — {GENE} in {SPECIES}")
print("=" * 70)

# ── 4a. Check for data ──────────────────────────────────────────────────
print("\n── 4a. Checking for transcriptome data ──")

data_available = False
adata = None

if trans_cfg.get("available"):
    geo_id = trans_cfg.get("geo_accession", "N/A")
    DATA_TRANS = PROJECT / 'data' / 'transcriptome' / cfg["species_slug"]
    DATA_TRANS.mkdir(parents=True, exist_ok=True)

    existing = list(DATA_TRANS.glob('*'))
    if existing:
        print(f"  Found files in {DATA_TRANS.name}:")
        for f in sorted(existing):
            size_mb = f.stat().st_size / 1e6
            print(f"    {f.name} ({size_mb:.1f} MB)")
        data_available = True
    else:
        print(f"  No data found in {DATA_TRANS}")
        print(f"  Reference: {trans_cfg.get('reference', 'N/A')}")
        if geo_id != "N/A":
            print(f"  GEO: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={geo_id}")

    # ── 4b. Try loading with scanpy ──
    print("\n── 4b. Attempting to load scRNA-seq data ──")
    scanpy_available = False
    try:
        import scanpy as sc
        scanpy_available = True
        print("  scanpy loaded successfully")
    except ImportError as e:
        print(f"  scanpy not available: {e}")

    if scanpy_available and data_available:
        h5ad_files = list(DATA_TRANS.glob('*.h5ad'))
        h5_files = list(DATA_TRANS.glob('*.h5'))

        if h5ad_files:
            adata = sc.read_h5ad(h5ad_files[0])
        elif h5_files:
            adata = sc.read_10x_h5(h5_files[0])

        if adata is not None:
            print(f"  Loaded: {adata.shape[0]} cells x {adata.shape[1]} genes")
            gene_names = list(adata.var_names)
            found_genes = []
            for term in [GENE, GENE.lower(), GENE.capitalize(), 'maestro', 'LOC']:
                matches = [g for g in gene_names if term.lower() in g.lower()]
                if matches:
                    found_genes.extend(matches[:5])
                    print(f"  Matches for '{term}': {matches[:5]}")
            found_genes = list(set(found_genes))

            if found_genes:
                if adata.X.max() > 100:
                    sc.pp.filter_cells(adata, min_genes=200)
                    sc.pp.filter_genes(adata, min_cells=3)
                    sc.pp.normalize_total(adata, target_sum=1e4)
                    sc.pp.log1p(adata)
                    sc.pp.pca(adata)
                    sc.pp.neighbors(adata)
                    sc.tl.umap(adata)

                fig, axes = plt.subplots(1, 2, figsize=(16, 6))
                sc.pl.umap(adata, color=found_genes[0], ax=axes[0], show=False,
                           title=f'{found_genes[0]} expression')
                region_col = None
                for col in ['brain_region', 'region', 'tissue', 'cluster', 'leiden']:
                    if col in adata.obs.columns:
                        region_col = col
                        break
                if region_col:
                    sc.pl.umap(adata, color=region_col, ax=axes[1], show=False)
                plt.tight_layout()
                plt.savefig(FIG_DIR / f'{GENE.lower()}_expression_umap.png',
                            dpi=150, bbox_inches='tight')
                plt.close()
                print(f"  Saved UMAP plot")
else:
    print(f"  No transcriptome data configured for {SPECIES}")
    print(f"  Skipping to illustrative analysis")

# ── 4c. Illustrative analysis ────────────────────────────────────────────
print("\n── 4c. Generating illustrative expression analysis ──")

regions = ['HVC', 'RA', 'Area X', 'LMAN', 'Cortex', 'Striatum', 'Cerebellum']
expected_expr = [0.82, 0.35, 0.51, 0.28, 0.12, 0.18, 0.05]
cell_types = ['Glutamatergic', 'GABAergic', 'Astrocyte', 'Oligodendrocyte', 'Microglia']
cell_expr = [0.65, 0.42, 0.08, 0.04, 0.02]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax = axes[0, 0]
colors_region = ['crimson' if e > 0.4 else ('darkorange' if e > 0.2 else 'steelblue')
                 for e in expected_expr]
bars = ax.bar(regions, expected_expr, color=colors_region, edgecolor='white')
ax.set_ylabel('Relative Expression')
ax.set_title(f'{GENE} Expected Expression by Brain Region\n(Illustrative — song nuclei highlighted)')
ax.tick_params(axis='x', rotation=30)
for bar, val in zip(bars, expected_expr):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{val:.2f}', ha='center', fontsize=9)

ax = axes[0, 1]
ax.barh(cell_types, cell_expr, color='steelblue', edgecolor='white')
ax.set_xlabel('Expression Level')
ax.set_title(f'{GENE} Expression by Cell Type (Illustrative)')

ax = axes[1, 0]
n_copies_expr = 20
np.random.seed(42)
copy_expr_vals = np.sort(np.random.exponential(0.3, n_copies_expr))[::-1]
copy_colors = ['crimson' if v > 0.5 else ('darkorange' if v > 0.2 else 'steelblue')
               for v in copy_expr_vals]
ax.bar(range(n_copies_expr), copy_expr_vals, color=copy_colors, edgecolor='white')
ax.set_xlabel(f'{GENE} Copy')
ax.set_ylabel('Expression Level')
ax.set_title('Per-copy expression (illustrative)')

ax = axes[1, 1]
song_nuclei = ['HVC', 'RA', 'Area X', 'LMAN']
non_song = ['Cortex', 'Striatum', 'Cerebellum']
song_mean = np.mean([expected_expr[regions.index(s)] for s in song_nuclei])
non_song_mean = np.mean([expected_expr[regions.index(s)] for s in non_song])
bars = ax.bar(['Song nuclei\n(HVC, RA, Area X, LMAN)',
               'Non-song regions\n(Cortex, Striatum, Cerebellum)'],
              [song_mean, non_song_mean], color=['crimson', 'steelblue'])
ax.set_ylabel('Mean Expression')
ax.set_title(f'Song vs Non-song: {song_mean/non_song_mean:.1f}x enrichment (illustrative)')

plt.suptitle(f'{GENE} Transcriptome Analysis — {SPECIES}', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(FIG_DIR / 'transcriptome_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {FIG_DIR / 'transcriptome_analysis.png'}")

# ── Summary ──
print("\n" + "=" * 70)
print(f"  TRANSCRIPTOME OVERLAY SUMMARY — {GENE} in {SPECIES}")
print("=" * 70)
if trans_cfg.get("available"):
    print(f"  Dataset: {trans_cfg.get('reference', 'N/A')}")
    print(f"  Data downloaded: {'Yes' if data_available else 'No'}")
else:
    print(f"  No transcriptome dataset configured for this species")
print(f"\n  => Proceed to Step 05")
print("=" * 70)
