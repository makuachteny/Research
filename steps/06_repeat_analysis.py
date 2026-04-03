#!/usr/bin/env python3
"""
06 — Repeat Structure Analysis
================================

QUESTION:
  What is the internal repeat structure of the MROH gene copies?
  How much of each copy is intact MROH sequence?
  What are the repeat lengths, and how do they compare across copies?

REASONING:
  MROH (Maestro Heat-like Repeat-containing) proteins are defined by
  tandem HEAT repeat motifs. Understanding the repeat content tells us:

    1. Copy integrity: What fraction of each gene unit is intact MROH?
       Copies with low coverage may be pseudogenised fragments.

    2. Repeat unit length: HEAT repeats are ~40 amino acids.
       Counting the number of internal repeats per copy reveals whether
       copies have expanded, contracted, or maintained repeat number.

    3. MROH content fraction: For each genomic locus, what percentage
       of the total nucleotide span is attributable to MROH coding
       sequence vs intergenic/intronic/repeat-element sequence?

    4. Internal repeat detection: Self-alignment of the reference protein
       reveals the tandem repeat architecture — critical for understanding
       which domains are under selection (from BEB analysis).

  If RepeatMasker is available, this step also runs it to characterise
  transposable element content (CR1 retrotransposons, LINEs, etc.) in
  the genomic regions flanking MROH copies.

Usage:
  python steps/06_repeat_analysis.py --species zebra_finch
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
from Bio import SeqIO
from pathlib import Path
import shutil
import subprocess

from species_config import load_config, get_data_dirs

parser = argparse.ArgumentParser(description='Step 06: Repeat Structure Analysis')
parser.add_argument('--species', required=True)
args, _ = parser.parse_known_args()

cfg = load_config(args.species)
dirs = get_data_dirs(cfg)

DATA_PROC = dirs["data_proc"]
FIG_DIR = dirs["fig_dir"]
TABLE_DIR = dirs["table_dir"]
prefix = cfg["output_prefix"]

GENE = cfg["gene_name"]
SPECIES = cfg["common_name"]

print("=" * 70)
print(f"  STEP 06: REPEAT STRUCTURE ANALYSIS — {GENE} in {SPECIES}")
print("=" * 70)


# ── 1. Load loci metadata and gene unit sequences ─────────────────────
print("\n── 1. Loading gene copy data ──")

loci_path = DATA_PROC / f'{prefix}_loci_table.csv'
gene_units_path = DATA_PROC / f'{prefix}_gene_units.fasta'

if not loci_path.exists() or not gene_units_path.exists():
    print(f"  ERROR: Required files not found. Run Step 01 first.")
    sys.exit(1)

loci_df = pd.read_csv(loci_path)
gene_units = {rec.id: rec for rec in SeqIO.parse(gene_units_path, 'fasta')}

print(f"  Loci in table: {len(loci_df)}")
print(f"  Gene unit sequences: {len(gene_units)}")

# ── 2. Load reference protein ─────────────────────────────────────────
print("\n── 2. Loading reference protein ──")

ref_protein_paths = cfg.get('reference_proteins', [])
ref_protein = None
ref_protein_len = 0

for rp in ref_protein_paths:
    rp_path = Path(rp)
    if not rp_path.is_absolute():
        rp_path = Path(__file__).resolve().parent.parent / rp
    if rp_path.exists():
        ref_protein = next(SeqIO.parse(rp_path, 'fasta'))
        ref_protein_len = len(ref_protein.seq)
        print(f"  Reference protein: {ref_protein.id}")
        print(f"  Reference length: {ref_protein_len} aa ({ref_protein_len * 3} nt CDS)")
        break

if ref_protein is None:
    print("  WARNING: No reference protein found. Using coverage data from loci table.")
    ref_protein_len = 643  # Default MROH6 length

# ── 3. Characterise copy lengths and MROH content ─────────────────────
print("\n── 3. Characterising copy lengths and MROH content ──")

# Each gene unit has: span (genomic), total_seq_len (CDS extracted), coverage_frac
loci_df['cds_length_bp'] = loci_df['total_seq_len']
loci_df['genomic_span_bp'] = loci_df['span']
loci_df['mroh_fraction'] = loci_df['coverage_frac'].clip(upper=1.0)

# Compute how much of the genomic span is CDS (MROH) vs non-coding
loci_df['mroh_nt_in_span'] = loci_df['cds_length_bp']
loci_df['non_mroh_nt'] = (loci_df['genomic_span_bp'] - loci_df['cds_length_bp']).clip(lower=0)
loci_df['pct_mroh_of_span'] = (loci_df['cds_length_bp'] / loci_df['genomic_span_bp'] * 100).clip(upper=100)

# Estimated number of HEAT repeats per copy
# HEAT repeat unit is ~40 amino acids = ~120 nt
HEAT_REPEAT_NT = 120
loci_df['est_heat_repeats'] = (loci_df['cds_length_bp'] / HEAT_REPEAT_NT).round(0).astype(int)

# Reference has ~15 HEAT repeats (643 aa / ~40 aa per repeat)
ref_heat_repeats = ref_protein_len / 40

n_copies = len(loci_df)
n_intact = len(loci_df[loci_df['coverage_frac'] >= 0.80])
n_partial = len(loci_df[(loci_df['coverage_frac'] >= 0.50) & (loci_df['coverage_frac'] < 0.80)])
n_fragment = len(loci_df[loci_df['coverage_frac'] < 0.50])

print(f"  Total gene copies: {n_copies}")
print(f"  Intact (>=80% cov): {n_intact} ({n_intact/n_copies*100:.1f}%)")
print(f"  Partial (50-80%):   {n_partial} ({n_partial/n_copies*100:.1f}%)")
print(f"  Fragment (<50%):    {n_fragment} ({n_fragment/n_copies*100:.1f}%)")
print()
print(f"  Genomic span — mean: {loci_df['genomic_span_bp'].mean():.0f} bp, "
      f"median: {loci_df['genomic_span_bp'].median():.0f} bp")
print(f"  CDS length   — mean: {loci_df['cds_length_bp'].mean():.0f} bp, "
      f"median: {loci_df['cds_length_bp'].median():.0f} bp")
print(f"  MROH fraction of span — mean: {loci_df['pct_mroh_of_span'].mean():.1f}%, "
      f"median: {loci_df['pct_mroh_of_span'].median():.1f}%")
print(f"  Estimated HEAT repeats — mean: {loci_df['est_heat_repeats'].mean():.1f}, "
      f"reference: ~{ref_heat_repeats:.0f}")

# ── 4. Internal repeat detection ──────────────────────────────────────
# WHY: HEAT repeats are ~38-47 amino acids and share structural similarity
# but LOW sequence identity (often <20%). Standard pairwise identity fails.
# Instead we use two complementary approaches:
#   (A) Compositional periodicity: HEAT repeats produce a periodic signal
#       in hydrophobicity because each unit alternates alpha-helices.
#   (B) Short k-mer enrichment: if the same short motifs recur at regular
#       intervals, the protein contains internal tandem repeats.
print("\n── 4. Internal repeat structure of reference protein ──")

ref_seq = str(ref_protein.seq) if ref_protein else ''
repeat_scores = []
HEAT_UNIT = 40  # canonical HEAT repeat length in amino acids

# Hydrophobicity scale (Kyte-Doolittle) — helical repeats show periodicity
HYDRO = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2,
}

if ref_seq:
    # --- Approach A: Hydrophobicity periodicity (autocorrelation) ---
    hydro_profile = np.array([HYDRO.get(aa, 0.0) for aa in ref_seq])
    # Smooth with window of 5 aa
    kernel = np.ones(5) / 5
    smoothed = np.convolve(hydro_profile, kernel, mode='same')

    # Autocorrelation to find repeat period
    n = len(smoothed)
    mean_h = np.mean(smoothed)
    var_h = np.var(smoothed)
    autocorr = []
    max_lag = min(100, n // 2)
    for lag in range(1, max_lag):
        c = np.mean((smoothed[:n-lag] - mean_h) * (smoothed[lag:] - mean_h))
        autocorr.append(c / var_h if var_h > 0 else 0)

    # Find peaks in autocorrelation — the dominant repeat period
    best_period = HEAT_UNIT
    if autocorr:
        # Look for peaks between 30 and 55 aa (HEAT repeat range)
        search_autocorr = autocorr[29:55] if len(autocorr) > 55 else autocorr[29:]
        if search_autocorr:
            best_period = 30 + np.argmax(search_autocorr)

    print(f"  Hydrophobicity autocorrelation peak: {best_period} aa")
    print(f"  (Expected HEAT repeat period: ~38-47 aa)")

    # --- Approach B: k-mer enrichment at periodic intervals ---
    kmer_size = 4
    kmer_positions = {}
    for i in range(len(ref_seq) - kmer_size + 1):
        kmer = ref_seq[i:i+kmer_size]
        kmer_positions.setdefault(kmer, []).append(i)

    # Find k-mers that recur at ~HEAT_UNIT intervals
    periodic_kmers = 0
    total_recurring = 0
    for kmer, positions in kmer_positions.items():
        if len(positions) < 3:
            continue
        total_recurring += 1
        spacings = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
        # Check if any spacing is near the HEAT repeat period (30-55 aa)
        near_period = sum(1 for s in spacings if 30 <= s <= 55)
        if near_period >= 1:
            periodic_kmers += 1

    periodicity_fraction = periodic_kmers / total_recurring if total_recurring > 0 else 0
    print(f"  Periodic 4-mers (spaced ~30-55 aa): {periodic_kmers}/{total_recurring} "
          f"({periodicity_fraction*100:.1f}%)")

    # --- Build per-position repeat score using local self-alignment ---
    # Use a shorter window (15 aa) and lower threshold (15% identity)
    # to catch the diverged HEAT repeats
    window = 15
    n_windows = len(ref_seq) - window + 1
    for i in range(n_windows):
        win_i = ref_seq[i:i+window]
        best_id = 0
        sum_id = 0
        n_nonself = 0
        for j in range(n_windows):
            if abs(i - j) < window:
                continue
            win_j = ref_seq[j:j+window]
            identity = sum(a == b for a, b in zip(win_i, win_j)) / window
            sum_id += identity
            n_nonself += 1
            if identity > best_id:
                best_id = identity
        mean_id = sum_id / n_nonself if n_nonself > 0 else 0
        repeat_scores.append({
            'position': i + 1,
            'repeat_score': mean_id,
            'best_match_id': best_id,
            'window_seq': win_i[:10] + '...'
        })

    repeat_score_df = pd.DataFrame(repeat_scores)
    scores = repeat_score_df['repeat_score'].values
    best_ids = repeat_score_df['best_match_id'].values

    # Threshold: positions where mean self-similarity exceeds background
    # Use 75th percentile as threshold — HEAT repeats are abundant
    threshold = np.percentile(scores, 75)
    repeat_regions = []
    in_repeat = False
    start = 0
    for i, s in enumerate(scores):
        if s > threshold and not in_repeat:
            start = i + 1
            in_repeat = True
        elif s <= threshold and in_repeat:
            if (i - start + 1) >= 15:  # Minimum repeat region size
                repeat_regions.append((start, i))
            in_repeat = False
    if in_repeat and (len(scores) - start + 1) >= 15:
        repeat_regions.append((start, len(scores)))

    # Estimate number of HEAT units by dividing repeat regions by period
    total_repeat_aa = sum(end - start + 1 for start, end in repeat_regions)
    pct_repeat = total_repeat_aa / ref_protein_len * 100
    est_n_units = total_repeat_aa / best_period if best_period > 0 else 0

    print(f"  Self-similarity window: {window} aa")
    print(f"  Mean repeat score: {np.mean(scores):.4f}")
    print(f"  Repeat threshold (75th %ile): {threshold:.4f}")
    print(f"  Repeat regions detected: {len(repeat_regions)}")
    for s, e in repeat_regions:
        print(f"    aa {s}-{e} ({e-s+1} aa, ~{(e-s+1)/best_period:.1f} HEAT units)")
    print(f"  Total repeat content: {total_repeat_aa} aa ({pct_repeat:.1f}% of protein)")
    print(f"  Estimated HEAT repeat units: ~{est_n_units:.0f}")

# ── 5. RepeatMasker (if available) ────────────────────────────────────
print("\n── 5. RepeatMasker analysis ──")

has_repeatmasker = shutil.which('RepeatMasker') is not None

if has_repeatmasker:
    print("  RepeatMasker found — running TE characterisation...")
    rm_dir = DATA_PROC / 'repeatmasker_output'
    rm_dir.mkdir(exist_ok=True)

    # Run RepeatMasker on gene unit sequences
    rm_cmd = [
        'RepeatMasker',
        '-species', 'aves',
        '-dir', str(rm_dir),
        '-pa', '4',
        '-gff',
        str(gene_units_path)
    ]
    try:
        result = subprocess.run(rm_cmd, capture_output=True, text=True, timeout=3600)
        rm_out = rm_dir / f'{gene_units_path.name}.out'
        if rm_out.exists():
            rm_results = []
            with open(rm_out) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('SW') or line.startswith('-'):
                        continue
                    parts = line.split()
                    if len(parts) >= 11:
                        try:
                            rm_results.append({
                                'query': parts[4],
                                'begin': int(parts[5]),
                                'end': int(parts[6]),
                                'repeat_class': parts[10],
                                'repeat_family': parts[9] if len(parts) > 9 else ''
                            })
                        except (ValueError, IndexError):
                            continue

            if rm_results:
                rm_df = pd.DataFrame(rm_results)
                print(f"  RepeatMasker hits: {len(rm_df)}")
                class_counts = rm_df['repeat_class'].value_counts()
                for cls, count in class_counts.head(10).items():
                    print(f"    {cls}: {count}")
                rm_df.to_csv(TABLE_DIR / 'repeatmasker_results.csv', index=False)
            else:
                print("  No repeat elements detected by RepeatMasker")
        else:
            print("  RepeatMasker ran but no output produced")
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"  RepeatMasker error: {e}")
else:
    print("  RepeatMasker not installed — using sequence-based analysis only")
    print("  To install: conda install -c bioconda repeatmasker")
    print("  (Continuing with CDS-based repeat characterisation)")

# ── 6. Figures ─────────────────────────────────────────────────────────
print("\n── 6. Generating figures ──")

sns.set_context('notebook')
sns.set_style('whitegrid')

# ── Figure 1: Copy length and MROH content overview ────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle(f'{GENE} Repeat Structure Analysis — {SPECIES}\n'
             f'({n_copies} gene copies)', fontsize=14, y=1.02)

# A: Genomic span distribution
ax = axes[0, 0]
ax.hist(loci_df['genomic_span_bp'], bins=40, color='steelblue', edgecolor='white', alpha=0.8)
ax.axvline(loci_df['genomic_span_bp'].median(), color='gold', linestyle='--',
           label=f'Median={loci_df["genomic_span_bp"].median():.0f} bp')
ax.set_xlabel('Genomic span (bp)')
ax.set_ylabel('Count')
ax.set_title('A. Genomic span per copy')
ax.legend(fontsize=8)

# B: CDS length distribution
ax = axes[0, 1]
ax.hist(loci_df['cds_length_bp'], bins=40, color='seagreen', edgecolor='white', alpha=0.8)
ref_cds = ref_protein_len * 3
ax.axvline(ref_cds, color='red', linestyle='--', label=f'Reference={ref_cds} bp')
ax.axvline(loci_df['cds_length_bp'].median(), color='gold', linestyle='--',
           label=f'Median={loci_df["cds_length_bp"].median():.0f} bp')
ax.set_xlabel('CDS length (bp)')
ax.set_ylabel('Count')
ax.set_title('B. CDS length per copy')
ax.legend(fontsize=8)

# C: Coverage fraction distribution
ax = axes[0, 2]
ax.hist(loci_df['coverage_frac'].clip(upper=1.5), bins=40, color='darkorange',
        edgecolor='white', alpha=0.8)
ax.axvline(0.80, color='green', linestyle='--', label='Intact (80%)')
ax.axvline(0.50, color='red', linestyle='--', label='Minimum (50%)')
ax.set_xlabel('Coverage fraction')
ax.set_ylabel('Count')
ax.set_title(f'C. MROH coverage: {n_intact} intact, {n_partial} partial, {n_fragment} fragment')
ax.legend(fontsize=8)

# D: MROH fraction of genomic span
ax = axes[1, 0]
ax.hist(loci_df['pct_mroh_of_span'], bins=40, color='purple', edgecolor='white', alpha=0.7)
ax.axvline(loci_df['pct_mroh_of_span'].median(), color='gold', linestyle='--',
           label=f'Median={loci_df["pct_mroh_of_span"].median():.1f}%')
ax.set_xlabel('% of genomic span that is MROH CDS')
ax.set_ylabel('Count')
ax.set_title('D. MROH content as fraction of total span')
ax.legend(fontsize=8)

# E: Estimated HEAT repeats per copy
ax = axes[1, 1]
ax.hist(loci_df['est_heat_repeats'], bins=range(0, int(loci_df['est_heat_repeats'].max()) + 3),
        color='teal', edgecolor='white', alpha=0.8)
ax.axvline(ref_heat_repeats, color='red', linestyle='--',
           label=f'Reference ~{ref_heat_repeats:.0f} repeats')
ax.set_xlabel('Estimated HEAT repeats')
ax.set_ylabel('Count')
ax.set_title('E. HEAT repeat units per copy')
ax.legend(fontsize=8)

# F: Copy integrity by chromosome class
ax = axes[1, 2]
if 'chrom_class' in loci_df.columns:
    class_summary = loci_df.groupby('chrom_class').agg(
        n_copies=('coverage_frac', 'count'),
        mean_cov=('coverage_frac', 'mean'),
        mean_span=('genomic_span_bp', 'mean')
    ).sort_values('n_copies', ascending=True)

    y_pos = range(len(class_summary))
    bars = ax.barh(y_pos, class_summary['mean_cov'], color='steelblue', alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{cls}\n(n={row['n_copies']:.0f})"
                        for cls, row in class_summary.iterrows()], fontsize=8)
    ax.axvline(0.80, color='green', linestyle='--', alpha=0.5)
    ax.axvline(0.50, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Mean coverage fraction')
    ax.set_title('F. Copy integrity by chromosome class')
    for i, (cls, row) in enumerate(class_summary.iterrows()):
        ax.text(row['mean_cov'] + 0.01, i, f'{row["mean_cov"]:.2f}', va='center', fontsize=8)

plt.tight_layout()
plt.savefig(FIG_DIR / 'repeat_structure.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {FIG_DIR / 'repeat_structure.png'}")

# ── Figure 2: Internal repeat structure of reference protein ───────
if repeat_scores:
    fig, axes = plt.subplots(3, 1, figsize=(16, 12),
                             gridspec_kw={'height_ratios': [2, 1.5, 2]})
    fig.suptitle(f'{GENE} Reference Protein Internal Repeat Structure — {SPECIES}',
                 fontsize=13, y=0.98)

    # A: Repeat similarity score along protein
    ax = axes[0]
    positions = [r['position'] for r in repeat_scores]
    scores_plot = [r['repeat_score'] for r in repeat_scores]
    best_plot = [r['best_match_id'] for r in repeat_scores]
    ax.fill_between(positions, scores_plot, alpha=0.3, color='steelblue', label='Mean self-similarity')
    ax.plot(positions, scores_plot, color='steelblue', linewidth=1)
    ax.plot(positions, best_plot, color='darkorange', linewidth=0.7, alpha=0.6, label='Best match identity')
    ax.axhline(threshold, color='red', linestyle='--', alpha=0.5,
               label=f'Repeat threshold (75th %ile)={threshold:.4f}')

    # Shade repeat regions
    for s, e in repeat_regions:
        ax.axvspan(s, e, alpha=0.15, color='crimson')

    # Mark exon boundaries
    exon_boundaries = cfg.get('exon_boundaries_aa', {})
    for exon_num, (start_aa, end_aa) in exon_boundaries.items():
        ax.axvline(start_aa, color='gray', linestyle=':', alpha=0.3, linewidth=0.5)

    ax.set_xlabel('Amino acid position')
    ax.set_ylabel('Similarity score')
    ax.set_title(f'A. Self-similarity profile ({window}-aa sliding window)\n'
                 f'{len(repeat_regions)} repeat regions, {total_repeat_aa} aa '
                 f'({pct_repeat:.1f}% of protein), ~{est_n_units:.0f} HEAT units')
    ax.legend(fontsize=8, loc='upper right')

    # B: Hydrophobicity autocorrelation — shows periodicity
    ax = axes[1]
    lags = list(range(1, len(autocorr) + 1))
    ax.bar(lags, autocorr, color='teal', alpha=0.6, width=1.0)
    ax.axvspan(30, 55, alpha=0.15, color='crimson', label='HEAT repeat range (30-55 aa)')
    ax.axvline(best_period, color='red', linestyle='--', linewidth=2,
               label=f'Best period = {best_period} aa')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Lag (amino acids)')
    ax.set_ylabel('Autocorrelation')
    ax.set_title(f'B. Hydrophobicity autocorrelation (repeat period detection)\n'
                 f'Peak at {best_period} aa — '
                 f'{"consistent with" if 30 <= best_period <= 55 else "outside"} HEAT repeat range')
    ax.legend(fontsize=8)
    ax.set_xlim(0, max_lag)

    # C: Dot plot (simplified — identity matrix)
    ax = axes[2]
    # Subsample for performance — use every other residue for proteins > 400 aa
    step = max(1, len(ref_seq) // 300)
    sub_seq = ref_seq[::step]
    n = len(sub_seq)
    dot_matrix = np.zeros((n, n))
    dot_window = 3
    for i in range(n - dot_window + 1):
        for j in range(i + dot_window, n - dot_window + 1):
            matches = sum(sub_seq[i+k] == sub_seq[j+k] for k in range(dot_window))
            if matches >= dot_window - 1:
                dot_matrix[i, j] = 1
                dot_matrix[j, i] = 1

    ax.imshow(dot_matrix, cmap='Blues', aspect='equal', interpolation='nearest')
    # Add diagonal lines at HEAT repeat intervals for reference
    for k in range(1, n // (best_period // step) + 1):
        offset = k * (best_period // step)
        if offset < n:
            ax.plot([0, n-offset], [offset, n], color='red', alpha=0.15, linewidth=0.5)
            ax.plot([offset, n], [0, n-offset], color='red', alpha=0.15, linewidth=0.5)

    ax.set_xlabel(f'Position (every {step} aa)')
    ax.set_ylabel(f'Position (every {step} aa)')
    ax.set_title(f'C. Self-dot-plot (diagonal lines = tandem repeats, '
                 f'red guides at {best_period}-aa intervals)')

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'repeat_internal_structure.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {FIG_DIR / 'repeat_internal_structure.png'}")

# ── 7. Save tables ─────────────────────────────────────────────────────
print("\n── 7. Saving results tables ──")

# Repeat summary per copy
repeat_cols = ['locus_id', 'chrom', 'chrom_class', 'genomic_span_bp', 'cds_length_bp',
               'coverage_frac', 'pct_mroh_of_span', 'est_heat_repeats']
available_cols = [c for c in repeat_cols if c in loci_df.columns]
loci_df[available_cols].to_csv(TABLE_DIR / 'repeat_structure.csv', index=False)
print(f"  Saved: {TABLE_DIR / 'repeat_structure.csv'}")

# Summary statistics
summary_rows = [
    {'Metric': 'Total gene copies', 'Value': n_copies},
    {'Metric': 'Intact copies (>=80% coverage)', 'Value': f'{n_intact} ({n_intact/n_copies*100:.1f}%)'},
    {'Metric': 'Partial copies (50-80%)', 'Value': f'{n_partial} ({n_partial/n_copies*100:.1f}%)'},
    {'Metric': 'Fragment copies (<50%)', 'Value': f'{n_fragment} ({n_fragment/n_copies*100:.1f}%)'},
    {'Metric': 'Reference protein length', 'Value': f'{ref_protein_len} aa'},
    {'Metric': 'Reference CDS length', 'Value': f'{ref_protein_len * 3} bp'},
    {'Metric': 'Mean genomic span', 'Value': f'{loci_df["genomic_span_bp"].mean():.0f} bp'},
    {'Metric': 'Median genomic span', 'Value': f'{loci_df["genomic_span_bp"].median():.0f} bp'},
    {'Metric': 'Mean CDS length', 'Value': f'{loci_df["cds_length_bp"].mean():.0f} bp'},
    {'Metric': 'Mean MROH fraction of span', 'Value': f'{loci_df["pct_mroh_of_span"].mean():.1f}%'},
    {'Metric': 'Estimated HEAT repeats (reference)', 'Value': f'~{ref_heat_repeats:.0f}'},
    {'Metric': 'Mean HEAT repeats per copy', 'Value': f'{loci_df["est_heat_repeats"].mean():.1f}'},
]

if repeat_scores:
    summary_rows.extend([
        {'Metric': 'Hydrophobicity autocorrelation peak', 'Value': f'{best_period} aa'},
        {'Metric': 'Internal repeat regions', 'Value': len(repeat_regions)},
        {'Metric': 'Total repeat content (reference)', 'Value': f'{total_repeat_aa} aa ({pct_repeat:.1f}%)'},
        {'Metric': 'Estimated HEAT units (reference)', 'Value': f'~{est_n_units:.0f}'},
        {'Metric': 'Periodic 4-mers', 'Value': f'{periodic_kmers}/{total_recurring} ({periodicity_fraction*100:.1f}%)'},
    ])

summary_rows.append({'Metric': 'RepeatMasker available', 'Value': 'Yes' if has_repeatmasker else 'No'})

pd.DataFrame(summary_rows).to_csv(TABLE_DIR / 'repeat_summary.csv', index=False)
print(f"  Saved: {TABLE_DIR / 'repeat_summary.csv'}")

# ── Summary ────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"  REPEAT STRUCTURE SUMMARY — {GENE} in {SPECIES}")
print("=" * 70)
print(f"  Total copies:              {n_copies}")
print(f"  Intact / Partial / Frag:   {n_intact} / {n_partial} / {n_fragment}")
print(f"  Reference protein:         {ref_protein_len} aa (~{ref_heat_repeats:.0f} HEAT repeats)")
print(f"  Mean CDS length:           {loci_df['cds_length_bp'].mean():.0f} bp")
print(f"  Mean MROH in genomic span: {loci_df['pct_mroh_of_span'].mean():.1f}%")
print(f"  Mean HEAT repeats/copy:    {loci_df['est_heat_repeats'].mean():.1f}")
if repeat_scores:
    print(f"  Internal repeat regions:   {len(repeat_regions)} ({pct_repeat:.1f}% of protein)")
print(f"  RepeatMasker:              {'Available' if has_repeatmasker else 'Not installed'}")
print("=" * 70)
