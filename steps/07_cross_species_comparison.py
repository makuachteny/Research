#!/usr/bin/env python3
"""
07 — Cross-Species Comparison Report
======================================

QUESTION:
  How do MROH gene copy dynamics compare across songbird species?
  Which species show the strongest evidence for RNA-mediated duplication
  and positive selection? Are the patterns consistent across lineages?

REASONING:
  Each species was analyzed independently through the same pipeline.
  Comparing the results reveals:

    1. Copy number variation — lineage-specific expansion rates
    2. Mutation rate elevation — strength of the RNA-mediated signal
    3. Selection regime — which species have positively selected sites
    4. Copy integrity — pseudogenization vs functional maintenance
    5. Repeat structure — conservation of HEAT repeat architecture

  A cross-species summary table and comparative figures enable direct
  hypothesis testing: if RNA-mediated duplication is a shared mechanism
  across songbirds, all species should show elevated mutation rates
  (>3x baseline) and dispersed genomic distribution.

Usage:
  python steps/07_cross_species_comparison.py
"""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent / 'scripts'))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from scipy import stats as sp_stats

PROJECT = Path(__file__).resolve().parent.parent
CONFIGS = PROJECT / 'configs'
RESULTS = PROJECT / 'results'
DATA_PROC = PROJECT / 'data' / 'processed'

FIG_DIR = RESULTS / 'figures'
TABLE_DIR = RESULTS / 'tables'
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("  STEP 07: CROSS-SPECIES COMPARISON REPORT")
print("=" * 70)

# ── 1. Load all species data ──────────────────────────────────────────
print("\n── 1. Loading results from all species ──")

species_slugs = sorted([p.stem for p in CONFIGS.glob('*.json')])
print(f"  Species found: {len(species_slugs)}")

all_data = []

from species_config import load_config

for slug in species_slugs:
    cfg = load_config(slug)

    row = {
        'species_slug': slug,
        'species_name': cfg['species_name'],
        'common_name': cfg['common_name'],
        'genome_assembly': cfg['genome_assembly'],
    }

    sp_results = RESULTS / slug / 'tables'
    sp_data = DATA_PROC / slug

    # Gene copy count
    loci_path = sp_data / f"{cfg['output_prefix']}_loci_table.csv"
    if loci_path.exists():
        loci_df = pd.read_csv(loci_path)
        row['n_copies'] = len(loci_df)
        row['n_intact'] = len(loci_df[loci_df['coverage_frac'] >= 0.80])
        row['n_partial'] = len(loci_df[(loci_df['coverage_frac'] >= 0.50) & (loci_df['coverage_frac'] < 0.80)])
        row['pct_intact'] = row['n_intact'] / row['n_copies'] * 100
        row['mean_coverage'] = loci_df['coverage_frac'].clip(upper=1.0).mean()
    else:
        row['n_copies'] = 0

    # Mutation rate summary
    mut_path = sp_results / 'mutation_rate_summary.csv'
    if mut_path.exists():
        mut_df = pd.read_csv(mut_path)
        mut_dict = dict(zip(mut_df.iloc[:, 0], mut_df.iloc[:, 1]))
        # Map various key names to standard names
        key_map = {
            'JC_mean': ['JC-corrected mean', 'JC_mean'],
            'JC_median': ['JC-corrected median', 'JC-corrected median divergence'],
            'TsTv_median': ['Ts/Tv median'],
            'fold_elevation': ['Fold elevation', 'fold_elevation'],
            'divergence_pvalue': ['P-value', 'T-test p-value (vs baseline)', 'P-value (vs baseline)'],
        }
        for target, candidates in key_map.items():
            for key in candidates:
                if key in mut_dict:
                    try:
                        row[target] = float(str(mut_dict[key]).replace('x', ''))
                    except (ValueError, TypeError):
                        pass
                    break

    # PAML results
    paml_path = sp_results / 'paml_results.csv'
    if paml_path.exists():
        paml_df = pd.read_csv(paml_path)
        for _, prow in paml_df.iterrows():
            model = prow.get('Model', '')
            if model == 'M0' and pd.notna(prow.get('omega')):
                row['M0_omega'] = prow['omega']
                row['M0_kappa'] = prow.get('kappa', np.nan)
            if model and pd.notna(prow.get('lnL')):
                row[f'{model}_lnL'] = prow['lnL']

        # Compute LRT
        if 'M1a_lnL' in row and 'M2a_lnL' in row:
            delta = 2 * (row['M2a_lnL'] - row['M1a_lnL'])
            row['LRT_M1a_vs_M2a'] = delta
            row['LRT_M1a_vs_M2a_p'] = sp_stats.chi2.sf(delta, 2) if delta > 0 else 1.0
        if 'M7_lnL' in row and 'M8_lnL' in row:
            delta = 2 * (row['M8_lnL'] - row['M7_lnL'])
            row['LRT_M7_vs_M8'] = delta
            row['LRT_M7_vs_M8_p'] = sp_stats.chi2.sf(delta, 2) if delta > 0 else 1.0

    # Pairwise dN/dS
    dnds_path = sp_results / 'pairwise_dnds.csv'
    if dnds_path.exists():
        dnds_df = pd.read_csv(dnds_path)
        valid = dnds_df.dropna(subset=['omega'])
        valid = valid[(valid['omega'] < 99) & (valid['dS'] > 0)]
        if len(valid) > 0:
            row['median_pairwise_omega'] = valid['omega'].median()
            row['mean_pairwise_dN'] = valid['dN'].mean()
            row['mean_pairwise_dS'] = valid['dS'].mean()
            row['n_pairs'] = len(valid)

    # Selection tests
    sel_path = sp_results / 'selection_tests.csv'
    if sel_path.exists():
        sel_df = pd.read_csv(sel_path)
        sel_dict = dict(zip(sel_df.iloc[:, 0], sel_df.iloc[:, 1]))
        try:
            row['BEB_sites_95'] = int(str(sel_dict.get('BEB sites (P>0.95)', 0)))
        except (ValueError, TypeError):
            row['BEB_sites_95'] = 0
        row['TsTv_4fold'] = sel_dict.get('Ts/Tv at 4-fold degenerate sites', '')

    # BEB selected sites
    beb_path = sp_results / 'beb_selected_sites.csv'
    if beb_path.exists():
        beb_df = pd.read_csv(beb_path)
        row['BEB_total'] = len(beb_df)
        row['BEB_sig_99'] = len(beb_df[beb_df['prob'] > 0.99]) if 'prob' in beb_df.columns else 0
        row['BEB_mean_omega'] = beb_df['omega'].mean() if 'omega' in beb_df.columns else np.nan
    else:
        row['BEB_total'] = 0
        row['BEB_sig_99'] = 0

    # Repeat summary
    rep_path = sp_results / 'repeat_summary.csv'
    if rep_path.exists():
        rep_df = pd.read_csv(rep_path)
        rep_dict = dict(zip(rep_df.iloc[:, 0], rep_df.iloc[:, 1]))
        try:
            row['mean_mroh_pct'] = float(str(rep_dict.get('Mean MROH fraction of span', '0')).replace('%', ''))
        except (ValueError, TypeError):
            pass
        try:
            row['mean_heat_repeats'] = float(rep_dict.get('Mean HEAT repeats per copy', 0))
        except (ValueError, TypeError):
            pass

    all_data.append(row)
    status = "FULL" if row.get('M0_omega') else "PARTIAL (no PAML)"
    print(f"  {slug:30s} — {row.get('n_copies', 0)} copies, {status}")

df = pd.DataFrame(all_data)

# ── 2. Summary table ──────────────────────────────────────────────────
print("\n── 2. Building cross-species comparison table ──")

summary_cols = {
    'common_name': 'Species',
    'n_copies': 'Gene Copies',
    'n_intact': 'Intact (>=80%)',
    'pct_intact': '% Intact',
    'JC_mean': 'Mean Divergence (JC)',
    'fold_elevation': 'Fold Elevation',
    'M0_omega': 'M0 omega (dN/dS)',
    'M0_kappa': 'M0 kappa (Ts/Tv)',
    'median_pairwise_omega': 'Median Pairwise omega',
    'BEB_total': 'BEB Sites (all)',
    'BEB_sites_95': 'BEB Sites (P>0.95)',
    'BEB_sig_99': 'BEB Sites (P>0.99)',
    'BEB_mean_omega': 'BEB Mean omega',
    'mean_mroh_pct': 'MROH % of Span',
    'mean_heat_repeats': 'HEAT Repeats/Copy',
}

available = {k: v for k, v in summary_cols.items() if k in df.columns}
summary = df[list(available.keys())].rename(columns=available)

# Round numeric columns
for col in summary.columns:
    if summary[col].dtype in ['float64', 'float32']:
        if 'pvalue' in col.lower() or 'p-value' in col.lower():
            summary[col] = summary[col].map(lambda x: f'{x:.2e}' if pd.notna(x) else '')
        elif '%' in col:
            summary[col] = summary[col].round(1)
        else:
            summary[col] = summary[col].round(3)

summary.to_csv(TABLE_DIR / 'cross_species_comparison.csv', index=False)
print(f"  Saved: {TABLE_DIR / 'cross_species_comparison.csv'}")

# Print table
print("\n  CROSS-SPECIES COMPARISON TABLE:")
print("  " + "─" * 80)
for col in summary.columns:
    vals = summary[col].astype(str).tolist()
    print(f"  {col:30s}  {'  |  '.join(vals)}")
print("  " + "─" * 80)

# ── 3. LRT significance table ────────────────────────────────────────
print("\n── 3. Likelihood ratio tests ──")

lrt_rows = []
for _, row in df.iterrows():
    lrt_row = {'Species': row['common_name']}
    for test, cols in [('M1a vs M2a', ('LRT_M1a_vs_M2a', 'LRT_M1a_vs_M2a_p')),
                       ('M7 vs M8', ('LRT_M7_vs_M8', 'LRT_M7_vs_M8_p'))]:
        delta = row.get(cols[0], np.nan)
        p = row.get(cols[1], np.nan)
        if pd.notna(delta):
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            lrt_row[f'{test} (2dlnL)'] = f'{delta:.2f}'
            lrt_row[f'{test} (p)'] = f'{p:.2e}'
            lrt_row[f'{test} (sig)'] = sig
        else:
            lrt_row[f'{test} (2dlnL)'] = 'N/A'
            lrt_row[f'{test} (p)'] = 'N/A'
            lrt_row[f'{test} (sig)'] = 'N/A'
    lrt_rows.append(lrt_row)

lrt_df = pd.DataFrame(lrt_rows)
lrt_df.to_csv(TABLE_DIR / 'cross_species_lrt.csv', index=False)
print(f"  Saved: {TABLE_DIR / 'cross_species_lrt.csv'}")

for _, row in lrt_df.iterrows():
    print(f"  {row['Species']:25s}  M1a/M2a: {row['M1a vs M2a (2dlnL)']:>8s} "
          f"(p={row['M1a vs M2a (p)']:>10s}) {row['M1a vs M2a (sig)']:>3s}  |  "
          f"M7/M8: {row['M7 vs M8 (2dlnL)']:>8s} "
          f"(p={row['M7 vs M8 (p)']:>10s}) {row['M7 vs M8 (sig)']:>3s}")

# ── 4. Figures ─────────────────────────────────────────────────────────
print("\n── 4. Generating cross-species comparison figures ──")

sns.set_context('notebook')
sns.set_style('whitegrid')

# Use species with data
species_order = df.sort_values('n_copies', ascending=False)['common_name'].tolist()
palette = sns.color_palette('Set2', n_colors=len(species_order))
color_map = dict(zip(species_order, palette))

# ── Figure 1: Multi-panel overview ─────────────────────────────────
fig = plt.figure(figsize=(20, 16))
gs = gridspec.GridSpec(3, 3, hspace=0.4, wspace=0.3)
fig.suptitle('MROH6 Cross-Species Comparison — 5 Songbird Species',
             fontsize=16, y=0.98, fontweight='bold')

# A: Gene copy number
ax = fig.add_subplot(gs[0, 0])
bars = ax.bar(df['common_name'], df['n_copies'], color=[color_map[n] for n in df['common_name']])
for bar, val in zip(bars, df['n_copies']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            str(int(val)), ha='center', fontsize=9, fontweight='bold')
ax.set_ylabel('Gene copies')
ax.set_title('A. MROH6 Copy Number')
ax.tick_params(axis='x', rotation=30)

# B: Copy integrity (stacked bar)
ax = fig.add_subplot(gs[0, 1])
intact_pct = df['pct_intact'].fillna(0).values
partial_pct = (df['n_partial'] / df['n_copies'] * 100).fillna(0).values
ax.bar(df['common_name'], intact_pct, color='seagreen', label='Intact (>=80%)')
ax.bar(df['common_name'], partial_pct, bottom=intact_pct, color='darkorange', label='Partial (50-80%)')
ax.set_ylabel('% of copies')
ax.set_title('B. Copy Integrity')
ax.legend(fontsize=8)
ax.tick_params(axis='x', rotation=30)

# C: Mutation rate elevation
ax = fig.add_subplot(gs[0, 2])
fold_vals = df['fold_elevation'].fillna(0).values
bars = ax.bar(df['common_name'], fold_vals, color=[color_map[n] for n in df['common_name']])
ax.axhline(3.0, color='red', linestyle='--', alpha=0.5, label='3x threshold (RNA signal)')
ax.axhline(1.0, color='gray', linestyle=':', alpha=0.5, label='Genomic baseline')
for bar, val in zip(bars, fold_vals):
    if val > 0:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1f}x', ha='center', fontsize=9, fontweight='bold')
ax.set_ylabel('Fold elevation vs baseline')
ax.set_title('C. Mutation Rate Elevation')
ax.legend(fontsize=8)
ax.tick_params(axis='x', rotation=30)

# D: M0 global omega
ax = fig.add_subplot(gs[1, 0])
has_m0 = df['M0_omega'].notna()
if has_m0.any():
    m0_df = df[has_m0]
    bars = ax.bar(m0_df['common_name'], m0_df['M0_omega'],
                  color=[color_map[n] for n in m0_df['common_name']])
    ax.axhline(1.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Neutral (omega=1)')
    ax.axhline(0.15, color='blue', linestyle=':', alpha=0.5, label='Bird avg (0.15)')
    for bar, val in zip(bars, m0_df['M0_omega']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                f'{val:.2f}', ha='center', fontsize=9, fontweight='bold')
    ax.legend(fontsize=8)
ax.set_ylabel('Global omega (dN/dS)')
ax.set_title('D. PAML M0 Global omega')
ax.tick_params(axis='x', rotation=30)

# E: BEB sites
ax = fig.add_subplot(gs[1, 1])
beb95 = df['BEB_sites_95'].fillna(0).values
beb99 = df['BEB_sig_99'].fillna(0).values
beb_other = df['BEB_total'].fillna(0).values - beb95
x = np.arange(len(df))
ax.bar(x, beb99, color='darkred', label='P>0.99 (**)')
ax.bar(x, beb95 - beb99, bottom=beb99, color='crimson', label='P>0.95 (*)')
ax.bar(x, beb_other, bottom=beb95, color='darkorange', alpha=0.6, label='Candidate (P<0.95)')
ax.set_xticks(x)
ax.set_xticklabels(df['common_name'], rotation=30, ha='right')
ax.set_ylabel('Number of BEB sites')
ax.set_title('E. Positively Selected Sites (BEB)')
ax.legend(fontsize=8)

# F: LRT significance
ax = fig.add_subplot(gs[1, 2])
has_lrt = df['LRT_M1a_vs_M2a'].notna()
if has_lrt.any():
    lrt_data = df[has_lrt]
    x = np.arange(len(lrt_data))
    w = 0.35
    ax.bar(x - w/2, lrt_data['LRT_M1a_vs_M2a'], w, color='steelblue', label='M1a vs M2a')
    ax.bar(x + w/2, lrt_data['LRT_M7_vs_M8'], w, color='darkorange', label='M7 vs M8')
    ax.axhline(5.99, color='red', linestyle='--', alpha=0.5, label='chi2 critical (p=0.05)')
    ax.axhline(9.21, color='darkred', linestyle='--', alpha=0.3, label='chi2 critical (p=0.01)')
    ax.set_xticks(x)
    ax.set_xticklabels(lrt_data['common_name'], rotation=30, ha='right')
    ax.legend(fontsize=7)
ax.set_ylabel('2*delta_lnL')
ax.set_title('F. Likelihood Ratio Tests')

# G: Median pairwise omega comparison
ax = fig.add_subplot(gs[2, 0])
has_pw = df['median_pairwise_omega'].notna()
if has_pw.any():
    pw_df = df[has_pw].sort_values('median_pairwise_omega', ascending=False)
    bars = ax.barh(pw_df['common_name'], pw_df['median_pairwise_omega'],
                   color=[color_map[n] for n in pw_df['common_name']])
    ax.axvline(0.15, color='blue', linestyle=':', alpha=0.5, label='Bird avg (0.15)')
    ax.axvline(0.40, color='darkorange', linestyle=':', alpha=0.5, label='Duplicated gene avg')
    ax.axvline(1.0, color='red', linestyle='--', alpha=0.3, label='Neutral')
    for bar, val in zip(bars, pw_df['median_pairwise_omega']):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9)
    ax.legend(fontsize=7, loc='lower right')
ax.set_xlabel('Median pairwise omega')
ax.set_title('G. Pairwise dN/dS Distribution')

# H: MROH content fraction
ax = fig.add_subplot(gs[2, 1])
mroh_pct = df['mean_mroh_pct'].fillna(0).values
bars = ax.bar(df['common_name'], mroh_pct, color=[color_map[n] for n in df['common_name']])
for bar, val in zip(bars, mroh_pct):
    if val > 0:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.0f}%', ha='center', fontsize=9, fontweight='bold')
ax.set_ylabel('Mean % of span that is MROH CDS')
ax.set_title('H. MROH Content Fraction')
ax.tick_params(axis='x', rotation=30)

# I: Divergence vs copy number scatter
ax = fig.add_subplot(gs[2, 2])
has_div = df['JC_mean'].notna()
if has_div.any():
    d = df[has_div]
    for _, row in d.iterrows():
        ax.scatter(row['n_copies'], row['JC_mean'],
                   c=[color_map[row['common_name']]], s=100, edgecolors='black',
                   linewidth=0.5, zorder=5)
        ax.annotate(row['common_name'], (row['n_copies'], row['JC_mean']),
                    xytext=(5, 5), textcoords='offset points', fontsize=7)
    # Add correlation if enough points
    if len(d) >= 3:
        r, p = sp_stats.pearsonr(d['n_copies'], d['JC_mean'])
        ax.set_title(f'I. Copy Number vs Divergence\n(r={r:.2f}, p={p:.3f})')
    else:
        ax.set_title('I. Copy Number vs Divergence')
    ax.axhline(0.03, color='blue', linestyle=':', alpha=0.5, label='Genomic baseline')
    ax.legend(fontsize=8)
ax.set_xlabel('Gene copies')
ax.set_ylabel('Mean JC-corrected divergence')

plt.savefig(FIG_DIR / 'cross_species_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {FIG_DIR / 'cross_species_comparison.png'}")

# ── Figure 2: Evidence summary heatmap ─────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))

# Build evidence matrix
evidence_cols = [
    ('Copies\n(n)', 'n_copies', 100, 600),
    ('Intact\n(%)', 'pct_intact', 0, 100),
    ('Divergence\n(JC)', 'JC_mean', 0, 0.6),
    ('Fold\nElevation', 'fold_elevation', 0, 20),
    ('M0\nomega', 'M0_omega', 0, 2),
    ('Pairwise\nomega', 'median_pairwise_omega', 0, 0.5),
    ('BEB\nSites', 'BEB_total', 0, 60),
    ('BEB\nomega', 'BEB_mean_omega', 0, 5),
    ('MROH\n(%)', 'mean_mroh_pct', 0, 70),
    ('HEAT\nRepeats', 'mean_heat_repeats', 0, 15),
]

evidence_matrix = np.zeros((len(df), len(evidence_cols)))
col_labels = []
for j, (label, col, vmin, vmax) in enumerate(evidence_cols):
    col_labels.append(label)
    if col in df.columns:
        vals = df[col].fillna(0).values
        # Normalize to 0-1
        if vmax > vmin:
            evidence_matrix[:, j] = np.clip((vals - vmin) / (vmax - vmin), 0, 1)

im = ax.imshow(evidence_matrix, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)

ax.set_xticks(range(len(col_labels)))
ax.set_xticklabels(col_labels, fontsize=9)
ax.set_yticks(range(len(df)))
ax.set_yticklabels([f"{row['common_name']}\n({row['species_name']})" for _, row in df.iterrows()],
                   fontsize=9)

# Add values
for i in range(len(df)):
    for j, (label, col, vmin, vmax) in enumerate(evidence_cols):
        val = df.iloc[i].get(col, 0)
        if pd.isna(val):
            txt = '—'
        elif isinstance(val, float):
            txt = f'{val:.2f}' if val < 10 else f'{val:.0f}'
        else:
            txt = str(int(val))
        color = 'white' if evidence_matrix[i, j] > 0.6 else 'black'
        ax.text(j, i, txt, ha='center', va='center', fontsize=8, color=color)

plt.colorbar(im, ax=ax, label='Normalized value', shrink=0.7)
ax.set_title('MROH6 Evidence Summary — All Species\n(Higher values = stronger signal)',
             fontsize=13, pad=15)
plt.tight_layout()
plt.savefig(FIG_DIR / 'cross_species_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {FIG_DIR / 'cross_species_heatmap.png'}")

# ── 5. Key findings ──────────────────────────────────────────────────
print("\n── 5. Key Findings ──")

# RNA-mediated duplication evidence
print("\n  RNA-MEDIATED DUPLICATION EVIDENCE:")
for _, row in df.iterrows():
    fold = row.get('fold_elevation', 0)
    if fold > 3:
        verdict = "STRONG (>3x baseline)"
    elif fold > 1:
        verdict = "Moderate"
    else:
        verdict = "Insufficient data"
    print(f"  {row['common_name']:25s}  {fold:>5.1f}x elevation — {verdict}")

# Positive selection evidence
print("\n  POSITIVE SELECTION EVIDENCE:")
for _, row in df.iterrows():
    m0 = row.get('M0_omega', np.nan)
    beb = row.get('BEB_total', 0)
    lrt_p = row.get('LRT_M1a_vs_M2a_p', np.nan)

    if pd.notna(m0) and pd.notna(lrt_p):
        if m0 > 1 and lrt_p < 0.05 and beb > 0:
            verdict = f"STRONG (omega={m0:.2f}, {beb} BEB sites, LRT p={lrt_p:.1e})"
        elif m0 > 1:
            verdict = f"Moderate (omega={m0:.2f}, LRT p={lrt_p:.1e})"
        else:
            verdict = f"Weak (omega={m0:.2f})"
    else:
        verdict = "N/A (PAML failed — gappy alignment)"
    print(f"  {row['common_name']:25s}  {verdict}")

# Consistency across species
n_elevated = sum(1 for _, r in df.iterrows() if r.get('fold_elevation', 0) > 3)
n_selection = sum(1 for _, r in df.iterrows()
                  if r.get('M0_omega', 0) > 1 and r.get('LRT_M1a_vs_M2a_p', 1) < 0.05)

print(f"\n  CONSISTENCY: {n_elevated}/{len(df)} species show >3x mutation rate elevation")
print(f"  CONSISTENCY: {n_selection}/{len(df)} species show significant positive selection (LRT)")

# ── Summary ────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  CROSS-SPECIES COMPARISON SUMMARY")
print("=" * 70)
print(f"  Species analyzed:          {len(df)}")
print(f"  Total gene copies:         {df['n_copies'].sum()}")
print(f"  Copy range:                {df['n_copies'].min()} — {df['n_copies'].max()}")
mean_fold = df['fold_elevation'].mean()
if pd.notna(mean_fold):
    print(f"  Mean fold elevation:       {mean_fold:.1f}x")
print(f"  RNA signal (>3x):          {n_elevated}/{len(df)} species")
print(f"  Positive selection (LRT):  {n_selection}/{len(df)} species")
print(f"\n  Reports: cross_species_comparison.csv, cross_species_lrt.csv")
print(f"  Figures: cross_species_comparison.png, cross_species_heatmap.png")
print("=" * 70)
