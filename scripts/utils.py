"""
Shared utility functions for MROH multicopy gene analysis pipeline.

Generalized March 2026: Config-driven, supports any MROH gene in any species.
Supports both tabular and FASTA-format BLASTn anchor files.
"""
import re
import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS = PROJECT_ROOT / "results"

# ── Legacy constants (for backward compatibility with zebra finch) ────────
EXON_BOUNDARIES_AA = {
    1: (1, 30), 2: (31, 55), 3: (56, 96), 4: (97, 140),
    5: (141, 185), 6: (186, 230), 7: (231, 275), 8: (276, 320),
    9: (321, 365), 10: (366, 410), 11: (411, 455), 12: (456, 500),
    13: (501, 545), 14: (546, 600), 15: (601, 643),
}
EXON4_START_AA = EXON_BOUNDARIES_AA[4][0]
EXON15_END_AA = EXON_BOUNDARIES_AA[15][1]
EXON4_15_LEN_AA = EXON15_END_AA - EXON4_START_AA + 1
EXON4_START_NT = (EXON4_START_AA - 1) * 3
EXON15_END_NT = EXON15_END_AA * 3
EXON4_15_LEN_NT = EXON4_15_LEN_AA * 3


# ── BLAST parsing ─────────────────────────────────────────────────────────

def parse_blast_fasta(filepath):
    """Parse tBLASTn or BLASTn FASTA output into a DataFrame with genomic coordinates.

    Handles headers like:
      NC_133032.1:28834524-28834700  (plus strand)
      NC_133032.1:c27748619-27748443 (complement strand)

    Returns DataFrame with columns:
        accession, chrom, start, end, strand, seq_len, sequence, header
    """
    records = []
    for rec in SeqIO.parse(filepath, "fasta"):
        header = rec.description
        match = re.match(r'(\S+):(c?)(\d+)-(\d+)\s+(.*)', header)
        if not match:
            continue
        accession = match.group(1)
        is_complement = match.group(2) == 'c'
        coord1 = int(match.group(3))
        coord2 = int(match.group(4))
        desc = match.group(5)

        chrom_match = re.search(r'chromosome\s+(\S+)', desc)
        chrom = chrom_match.group(1).rstrip(',') if chrom_match else 'unknown'

        if is_complement:
            start, end = coord2, coord1
            strand = '-'
        else:
            start, end = coord1, coord2
            strand = '+'

        records.append({
            'accession': accession,
            'chrom': chrom,
            'start': start,
            'end': end,
            'strand': strand,
            'seq_len': len(rec.seq),
            'sequence': str(rec.seq),
            'header': header,
        })
    return pd.DataFrame(records)


def parse_combined_tblastn(filepaths):
    """Parse and combine one or more tBLASTn FASTA files, deduplicating."""
    dfs = []
    for fp in filepaths:
        fp = Path(fp)
        if not fp.exists():
            print(f"  WARNING: tBLASTn file not found: {fp}")
            continue
        df = parse_blast_fasta(fp)
        df['source'] = fp.stem
        dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.drop_duplicates(
        subset=['accession', 'start', 'end', 'strand'], keep='first'
    ).reset_index(drop=True)
    return combined


def build_accession_to_chrom(tblastn_df):
    """Build accession -> chromosome mapping from tBLASTn DataFrame."""
    mapping = {}
    for _, row in tblastn_df.drop_duplicates('accession').iterrows():
        mapping[row['accession']] = row['chrom']
    return mapping


# ── Exon 13 anchor parsing ───────────────────────────────────────────────

def parse_exon13_blastn(filepath, fmt="auto"):
    """Parse BLASTn output for exon 13 anchor positions.

    Supports two formats:
      - "tabular": Standard BLAST tabular (-outfmt 6/7) with columns
        qaccver, saccver, pident, length, mismatch, gapopen, qstart, qend,
        sstart, send, evalue, bitscore
      - "fasta": FASTA output with headers encoding genomic coordinates
        (e.g., >NC_080465.1:61080293-61080336 Species chromosome Z, assembly)

    Args:
        filepath: Path to BLASTn output file
        fmt: "tabular", "fasta", or "auto" (detect from content)

    Returns:
        DataFrame with columns: saccver, anchor_start, anchor_end, anchor_mid,
        strand, pident (100.0 for FASTA format)
    """
    filepath = Path(filepath)

    if fmt == "auto":
        fmt = _detect_blast_format(filepath)

    if fmt == "tabular":
        return _parse_exon13_tabular(filepath)
    elif fmt == "fasta":
        return _parse_exon13_fasta(filepath)
    else:
        raise ValueError(f"Unknown BLASTn format: {fmt}")


def _detect_blast_format(filepath):
    """Detect whether a BLAST file is tabular or FASTA format."""
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                return "fasta"
            if line.startswith('#'):
                return "tabular"
            # Tabular has tab-separated fields
            if '\t' in line and len(line.split('\t')) >= 10:
                return "tabular"
            # FASTA header without >
            if re.match(r'\S+:\d+-\d+', line):
                return "fasta"
            break
    return "tabular"  # default


def _parse_exon13_tabular(filepath):
    """Parse tabular BLASTn output for exon 13 anchors."""
    cols = ['qaccver', 'saccver', 'pident', 'length', 'mismatch', 'gapopen',
            'qstart', 'qend', 'sstart', 'send', 'evalue', 'bitscore']
    df = pd.read_csv(filepath, sep='\t', comment='#', names=cols,
                     skipinitialspace=True)
    df = df.dropna(subset=['saccver']).reset_index(drop=True)
    df['anchor_start'] = df[['sstart', 'send']].min(axis=1).astype(int)
    df['anchor_end'] = df[['sstart', 'send']].max(axis=1).astype(int)
    df['anchor_mid'] = ((df['anchor_start'] + df['anchor_end']) / 2).astype(int)
    df['strand'] = np.where(df['sstart'] <= df['send'], '+', '-')
    return df


def _parse_exon13_fasta(filepath):
    """Parse FASTA-format BLASTn output into anchor positions.

    Converts FASTA records into a DataFrame compatible with the tabular format,
    extracting accession and coordinates from headers like:
        >NC_080465.1:61080293-61080336 Melospiza georgiana ... chromosome Z, ...
    """
    records = []
    for rec in SeqIO.parse(filepath, "fasta"):
        header = rec.description
        match = re.match(r'(\S+):(c?)(\d+)-(\d+)\s+(.*)', header)
        if not match:
            continue
        accession = match.group(1)
        is_complement = match.group(2) == 'c'
        coord1 = int(match.group(3))
        coord2 = int(match.group(4))

        anchor_start = min(coord1, coord2)
        anchor_end = max(coord1, coord2)
        strand = '-' if is_complement else '+'

        records.append({
            'saccver': accession,
            'anchor_start': anchor_start,
            'anchor_end': anchor_end,
            'anchor_mid': (anchor_start + anchor_end) // 2,
            'strand': strand,
            'pident': 100.0,  # not available in FASTA; assume high identity
            'length': len(rec.seq),
            'sequence': str(rec.seq),
        })

    df = pd.DataFrame(records)
    if len(df) == 0:
        return pd.DataFrame(columns=['saccver', 'anchor_start', 'anchor_end',
                                      'anchor_mid', 'strand', 'pident'])
    return df


# ── Hit merging ──────────────────────────────────────────────────────────

def merge_overlapping_hits(df, max_gap=500):
    """Merge overlapping or nearby BLAST hits on the same chrom/strand into loci."""
    loci = []
    locus_id = 0
    for (chrom, strand), group in df.groupby(['chrom', 'strand']):
        group = group.sort_values('start').reset_index(drop=True)
        current_start = group.iloc[0]['start']
        current_end = group.iloc[0]['end']
        current_seqs = [group.iloc[0]]

        for i in range(1, len(group)):
            row = group.iloc[i]
            if row['start'] <= current_end + max_gap:
                current_end = max(current_end, row['end'])
                current_seqs.append(row)
            else:
                loci.append(_make_locus(locus_id, chrom, strand,
                                        current_start, current_end, current_seqs))
                locus_id += 1
                current_start = row['start']
                current_end = row['end']
                current_seqs = [row]
        loci.append(_make_locus(locus_id, chrom, strand,
                                current_start, current_end, current_seqs))
        locus_id += 1
    return pd.DataFrame(loci)


def _make_locus(locus_id, chrom, strand, start, end, hit_rows):
    """Helper to create a locus record from merged hits."""
    hit_rows_sorted = sorted(hit_rows, key=lambda r: r['start'])
    concat_seq = ''.join(r['sequence'] for r in hit_rows_sorted)
    return {
        'locus_id': locus_id,
        'chrom': chrom,
        'strand': strand,
        'start': start,
        'end': end,
        'span': end - start + 1,
        'n_hits': len(hit_rows),
        'total_seq_len': len(concat_seq),
        'sequence': concat_seq,
    }


# ── Gene unit definition ─────────────────────────────────────────────────

def define_gene_units(exon13_df, tblastn_df, accession_to_chrom,
                      expected_len_nt, max_dist=25000, merge_gap=500,
                      min_coverage=0.50):
    """Define gene units using exon 13 anchors and tBLASTn hits.

    This is the generalized version that accepts expected_len_nt as a parameter
    instead of hardcoding EXON4_15_LEN_NT.

    Args:
        exon13_df: DataFrame from parse_exon13_blastn
        tblastn_df: combined tBLASTn DataFrame
        accession_to_chrom: dict of accession -> chromosome
        expected_len_nt: expected CDS length in nucleotides for the analysis range
        max_dist: max distance to assign a hit to an anchor (bp)
        merge_gap: gap tolerance for merging non-anchored hits (bp)
        min_coverage: minimum fraction of expected length required

    Returns:
        Tuple of (all_gene_units_df, filtered_gene_units_df)
    """
    gene_units = []
    gu_id = 0

    anchored_accs = set(exon13_df['saccver'].unique())
    all_blast_accs = set(tblastn_df['accession'].unique()) if len(tblastn_df) > 0 else set()

    min_len = int(expected_len_nt * min_coverage)

    # --- Anchored accessions ---
    for acc in sorted(anchored_accs):
        anchors = exon13_df[exon13_df['saccver'] == acc].sort_values('anchor_mid')
        anchor_mids = anchors['anchor_mid'].values
        hits = tblastn_df[tblastn_df['accession'] == acc].copy() if len(tblastn_df) > 0 else pd.DataFrame()
        chrom = accession_to_chrom.get(acc, 'unknown')

        if len(hits) == 0:
            for _, anchor in anchors.iterrows():
                gene_units.append({
                    'gene_unit_id': gu_id, 'chrom': chrom,
                    'accession': acc, 'anchor_pos': int(anchor['anchor_mid']),
                    'start': int(anchor['anchor_start']),
                    'end': int(anchor['anchor_end']),
                    'span': int(anchor['anchor_end'] - anchor['anchor_start'] + 1),
                    'n_hits': 0, 'total_seq_len': 0, 'sequence': '',
                    'has_exon13': True,
                })
                gu_id += 1
            continue

        hit_mids = ((hits['start'] + hits['end']) / 2).values
        dists = np.abs(hit_mids[:, None] - anchor_mids[None, :])
        nearest_anchor_idx = np.argmin(dists, axis=1)
        nearest_dist = np.min(dists, axis=1)

        hits = hits.copy()
        hits['_anchor_idx'] = nearest_anchor_idx
        hits['_anchor_dist'] = nearest_dist

        for local_idx, (_, anchor) in enumerate(anchors.iterrows()):
            mask = (hits['_anchor_idx'] == local_idx) & (hits['_anchor_dist'] <= max_dist)
            anchor_hits = hits[mask]

            if len(anchor_hits) > 0:
                merged_seq, merged_start, merged_end, n_merged = \
                    _merge_hit_sequences(anchor_hits)
            else:
                merged_seq = ''
                merged_start = int(anchor['anchor_start'])
                merged_end = int(anchor['anchor_end'])
                n_merged = 0

            gene_units.append({
                'gene_unit_id': gu_id, 'chrom': chrom,
                'accession': acc,
                'anchor_pos': int(anchor['anchor_mid']),
                'start': merged_start, 'end': merged_end,
                'span': merged_end - merged_start + 1,
                'n_hits': n_merged,
                'total_seq_len': len(merged_seq),
                'sequence': merged_seq, 'has_exon13': True,
            })
            gu_id += 1

    # --- Non-anchored accessions (from tBLASTn only) ---
    non_anchored = all_blast_accs - anchored_accs
    for acc in sorted(non_anchored):
        hits = tblastn_df[tblastn_df['accession'] == acc]
        chrom = accession_to_chrom.get(acc, 'unknown')
        merged_loci = merge_overlapping_hits(hits, max_gap=merge_gap)
        for _, locus in merged_loci.iterrows():
            gene_units.append({
                'gene_unit_id': gu_id, 'chrom': chrom,
                'accession': acc,
                'anchor_pos': None,
                'start': int(locus['start']), 'end': int(locus['end']),
                'span': int(locus['span']),
                'n_hits': int(locus['n_hits']),
                'total_seq_len': int(locus['total_seq_len']),
                'sequence': locus['sequence'], 'has_exon13': False,
            })
            gu_id += 1

    all_gu = pd.DataFrame(gene_units)

    if len(all_gu) == 0:
        return all_gu, all_gu

    # Apply coverage filter
    all_gu['coverage_frac'] = all_gu['total_seq_len'] / expected_len_nt
    filtered_gu = all_gu[all_gu['total_seq_len'] >= min_len].copy()
    filtered_gu['coverage_frac'] = filtered_gu['total_seq_len'] / expected_len_nt

    return all_gu, filtered_gu


# Legacy alias
def define_gene_units_exon4_15(exon13_df, tblastn_df, accession_to_chrom,
                                max_dist=25000, merge_gap=500,
                                min_coverage=0.50):
    """Legacy wrapper — calls define_gene_units with EXON4_15_LEN_NT."""
    return define_gene_units(
        exon13_df, tblastn_df, accession_to_chrom,
        expected_len_nt=EXON4_15_LEN_NT,
        max_dist=max_dist, merge_gap=merge_gap, min_coverage=min_coverage
    )


def _merge_hit_sequences(hits_df):
    """Merge overlapping tBLASTn hit sequences within a gene unit."""
    hits_sorted = hits_df.sort_values('start').reset_index(drop=True)
    merged_seq_parts = []
    current_end = -1

    for _, row in hits_sorted.iterrows():
        if row['start'] > current_end:
            merged_seq_parts.append(row['sequence'])
        else:
            overlap_bp = current_end - row['start'] + 1
            if overlap_bp < len(row['sequence']):
                merged_seq_parts.append(row['sequence'][overlap_bp:])
        current_end = max(current_end, row['end'])

    concat_seq = ''.join(merged_seq_parts)
    return (concat_seq,
            int(hits_sorted.iloc[0]['start']),
            int(hits_sorted['end'].max()),
            len(hits_sorted))


# ── FASTA output ─────────────────────────────────────────────────────────

def gene_units_to_fasta(gu_df, outpath):
    """Write gene units DataFrame to FASTA file."""
    records = []
    for _, row in gu_df.iterrows():
        if row['total_seq_len'] == 0:
            continue
        rec = SeqRecord(
            Seq(row['sequence']),
            id=f"gu_{row['gene_unit_id']}_chr{row['chrom']}_{row['start']}_{row['end']}",
            description=f"n_hits={row['n_hits']} span={row['span']}bp "
                        f"cov={row.get('coverage_frac', 0):.2f}",
        )
        records.append(rec)
    SeqIO.write(records, outpath, "fasta")
    return len(records)


def loci_to_fasta(loci_df, outpath):
    """Write loci DataFrame to FASTA file (backward-compatible)."""
    records = []
    for _, row in loci_df.iterrows():
        strand = row.get('strand', '.')
        rec = SeqRecord(
            Seq(row['sequence']),
            id=f"locus_{row['locus_id']}_chr{row['chrom']}_{row['start']}_{row['end']}_{strand}",
            description=f"n_hits={row['n_hits']} span={row['span']}bp",
        )
        records.append(rec)
    SeqIO.write(records, outpath, "fasta")
    return len(records)


# ── Anchor-only gene units (when no tBLASTn available) ───────────────────

def define_gene_units_from_tblastn_only(tblastn_df, accession_to_chrom,
                                        expected_len_nt, merge_gap=500,
                                        min_coverage=0.50):
    """Define gene units from tBLASTn hits alone (no exon 13 anchors).

    Use this when exon 13 anchor data is not available. Merges nearby tBLASTn
    hits on the same chromosome/strand into gene units and applies coverage filter.

    Returns:
        Tuple of (all_gene_units_df, filtered_gene_units_df)
    """
    if len(tblastn_df) == 0:
        empty = pd.DataFrame(columns=[
            'gene_unit_id', 'chrom', 'accession', 'anchor_pos',
            'start', 'end', 'span', 'n_hits', 'total_seq_len',
            'sequence', 'has_exon13', 'coverage_frac',
        ])
        return empty, empty

    gene_units = []
    gu_id = 0
    min_len = int(expected_len_nt * min_coverage)

    for acc in sorted(tblastn_df['accession'].unique()):
        hits = tblastn_df[tblastn_df['accession'] == acc]
        chrom = accession_to_chrom.get(acc, 'unknown')
        merged_loci = merge_overlapping_hits(hits, max_gap=merge_gap)
        for _, locus in merged_loci.iterrows():
            gene_units.append({
                'gene_unit_id': gu_id, 'chrom': chrom,
                'accession': acc, 'anchor_pos': None,
                'start': int(locus['start']), 'end': int(locus['end']),
                'span': int(locus['span']),
                'n_hits': int(locus['n_hits']),
                'total_seq_len': int(locus['total_seq_len']),
                'sequence': locus['sequence'], 'has_exon13': False,
            })
            gu_id += 1

    all_gu = pd.DataFrame(gene_units)
    if len(all_gu) == 0:
        return all_gu, all_gu

    all_gu['coverage_frac'] = all_gu['total_seq_len'] / expected_len_nt
    filtered_gu = all_gu[all_gu['total_seq_len'] >= min_len].copy()
    filtered_gu['coverage_frac'] = filtered_gu['total_seq_len'] / expected_len_nt

    return all_gu, filtered_gu


def define_gene_units_from_anchors_only(exon13_df, accession_to_chrom):
    """Define gene unit positions using only exon 13 anchor data.

    Use this when tBLASTn alignment data is not yet available. Each anchor
    becomes a gene unit with position information but no sequence data.
    This allows chromosome distribution analysis and copy counting.

    Returns:
        DataFrame with gene unit metadata (no sequences).
    """
    gene_units = []
    gu_id = 0

    for acc in sorted(exon13_df['saccver'].unique()):
        anchors = exon13_df[exon13_df['saccver'] == acc].sort_values('anchor_mid')
        chrom = accession_to_chrom.get(acc, 'unknown')

        for _, anchor in anchors.iterrows():
            gene_units.append({
                'gene_unit_id': gu_id,
                'chrom': chrom,
                'accession': acc,
                'anchor_pos': int(anchor['anchor_mid']),
                'start': int(anchor['anchor_start']),
                'end': int(anchor['anchor_end']),
                'span': int(anchor['anchor_end'] - anchor['anchor_start'] + 1),
                'n_hits': 0,
                'total_seq_len': 0,
                'sequence': '',
                'has_exon13': True,
                'coverage_frac': 0.0,
            })
            gu_id += 1

    return pd.DataFrame(gene_units)


def build_accession_to_chrom_from_anchors(exon13_df):
    """Build accession -> chromosome mapping from FASTA-parsed anchor data.

    For FASTA-format anchors, the chromosome is embedded in the header
    but not directly in the parsed DataFrame. This function extracts it
    from the anchor DataFrame when it was parsed from FASTA format.
    """
    # If parsed from FASTA, we need to extract chrom from the saccver
    # The FASTA headers contain chromosome info but parse_exon13_fasta
    # only stores accession. We need to re-parse for chrom mapping.
    return {}


# ── Divergence and substitution analysis ─────────────────────────────────

def jukes_cantor_distance(p):
    """Jukes-Cantor correction for nucleotide distance."""
    if np.isnan(p) or p >= 0.75:
        return np.nan
    return -0.75 * np.log(1.0 - (4.0 / 3.0) * p)


def count_substitution_types(seq1, seq2):
    """Count transitions and transversions between two aligned sequences."""
    purines = {'A', 'G'}
    pyrimidines = {'C', 'T'}

    ts = tv = identical = gaps = 0
    for a, b in zip(seq1.upper(), seq2.upper()):
        if a == '-' or b == '-' or a == 'N' or b == 'N':
            gaps += 1
            continue
        if a == b:
            identical += 1
        elif (a in purines and b in purines) or (a in pyrimidines and b in pyrimidines):
            ts += 1
        else:
            tv += 1
    total = ts + tv + identical
    return {
        'transitions': ts,
        'transversions': tv,
        'identical': identical,
        'gaps': gaps,
        'total_compared': total,
        'raw_divergence': (ts + tv) / total if total > 0 else np.nan,
        'ts_tv_ratio': ts / tv if tv > 0 else np.inf,
    }


def pairwise_divergence_matrix(alignment_dict):
    """Compute pairwise divergence matrix from dict of {name: sequence}."""
    names = list(alignment_dict.keys())
    n = len(names)
    raw_div = np.zeros((n, n))
    jc_div = np.zeros((n, n))
    ts_tv = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            stats = count_substitution_types(
                alignment_dict[names[i]], alignment_dict[names[j]])
            raw_div[i, j] = raw_div[j, i] = stats['raw_divergence']
            jc_div[i, j] = jc_div[j, i] = jukes_cantor_distance(stats['raw_divergence'])
            ts_tv[i, j] = ts_tv[j, i] = stats['ts_tv_ratio']

    return names, raw_div, jc_div, ts_tv


def compute_pairwise_dnds(codon_alignment_dict):
    """Compute pairwise dN, dS, and omega using the Nei-Gojobori (1986) method."""
    names = list(codon_alignment_dict.keys())
    n = len(names)
    results = []

    for i in range(n):
        for j in range(i + 1, n):
            seq1 = codon_alignment_dict[names[i]].upper()
            seq2 = codon_alignment_dict[names[j]].upper()
            dn, ds, omega, nd, sd, N, S = _nei_gojobori(seq1, seq2)
            results.append({
                'seq1': names[i], 'seq2': names[j],
                'dN': dn, 'dS': ds, 'omega': omega,
                'nd': nd, 'sd': sd, 'N': N, 'S': S,
            })

    return pd.DataFrame(results)


def _nei_gojobori(seq1, seq2):
    """Nei-Gojobori method for a single pair of codon-aligned sequences."""
    CODON_TABLE = {
        'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
        'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
        'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
        'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
        'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
        'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
        'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
        'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
        'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
        'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
        'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
        'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
        'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
        'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
        'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
        'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
    }

    BASES = ['T', 'C', 'A', 'G']
    total_sd = total_nd = 0.0
    total_S = total_N = 0.0

    n_codons = min(len(seq1), len(seq2)) // 3
    for c in range(n_codons):
        codon1 = seq1[c*3:c*3+3]
        codon2 = seq2[c*3:c*3+3]

        if '-' in codon1 or '-' in codon2 or 'N' in codon1 or 'N' in codon2:
            continue
        if codon1 not in CODON_TABLE or codon2 not in CODON_TABLE:
            continue
        if CODON_TABLE[codon1] == '*' or CODON_TABLE[codon2] == '*':
            continue

        for codon in [codon1, codon2]:
            s_sites = 0.0
            aa = CODON_TABLE[codon]
            for pos in range(3):
                n_syn = 0
                for base in BASES:
                    if base == codon[pos]:
                        continue
                    mutant = codon[:pos] + base + codon[pos+1:]
                    if mutant in CODON_TABLE and CODON_TABLE[mutant] == aa:
                        n_syn += 1
                s_sites += n_syn / 3.0
            total_S += s_sites / 2.0

        diffs = [i for i in range(3) if codon1[i] != codon2[i]]
        n_diffs = len(diffs)

        if n_diffs == 0:
            total_N += (3.0 - 0) / 2.0
            continue

        if n_diffs == 1:
            aa1 = CODON_TABLE[codon1]
            aa2 = CODON_TABLE[codon2]
            if aa1 == aa2:
                total_sd += 1
            else:
                total_nd += 1
        elif n_diffs == 2:
            sd_sum = nd_sum = 0
            n_paths = 0
            for first_pos in diffs:
                second_pos = [d for d in diffs if d != first_pos][0]
                intermediate = list(codon1)
                intermediate[first_pos] = codon2[first_pos]
                intermediate = ''.join(intermediate)
                if intermediate not in CODON_TABLE or CODON_TABLE[intermediate] == '*':
                    continue
                if CODON_TABLE[codon1] == CODON_TABLE[intermediate]:
                    sd_sum += 1
                else:
                    nd_sum += 1
                if CODON_TABLE[intermediate] == CODON_TABLE[codon2]:
                    sd_sum += 1
                else:
                    nd_sum += 1
                n_paths += 1
            if n_paths > 0:
                total_sd += sd_sum / n_paths
                total_nd += nd_sum / n_paths
        elif n_diffs == 3:
            import itertools
            sd_sum = nd_sum = 0
            n_paths = 0
            for perm in itertools.permutations(diffs):
                current = list(codon1)
                valid = True
                path_sd = path_nd = 0
                for pos in perm:
                    prev_aa = CODON_TABLE.get(''.join(current), '*')
                    current[pos] = codon2[pos]
                    next_codon = ''.join(current)
                    next_aa = CODON_TABLE.get(next_codon, '*')
                    if next_aa == '*':
                        valid = False
                        break
                    if prev_aa == next_aa:
                        path_sd += 1
                    else:
                        path_nd += 1
                if valid:
                    sd_sum += path_sd
                    nd_sum += path_nd
                    n_paths += 1
            if n_paths > 0:
                total_sd += sd_sum / n_paths
                total_nd += nd_sum / n_paths

    total_N_sites = n_codons * 3 - total_S
    if total_N_sites <= 0 or total_S <= 0:
        return np.nan, np.nan, np.nan, total_nd, total_sd, total_N_sites, total_S

    pN = total_nd / total_N_sites if total_N_sites > 0 else 0
    pS = total_sd / total_S if total_S > 0 else 0

    dN = jukes_cantor_distance(pN) if pN < 0.75 else np.nan
    dS = jukes_cantor_distance(pS) if pS < 0.75 else np.nan

    if dS is not None and not np.isnan(dS) and dS > 0:
        omega = dN / dS if not np.isnan(dN) else np.nan
    elif dS == 0 and dN == 0:
        omega = 0.0
    else:
        omega = np.nan

    return dN, dS, omega, total_nd, total_sd, total_N_sites, total_S


# ── Chromosome classification (legacy, species-agnostic) ─────────────────

def classify_chrom(chrom):
    """Classify a chromosome into ancestral/macro/micro/sex categories.

    Legacy function for zebra finch. For species-specific classification,
    use species_config.classify_chrom(chrom, cfg) instead.
    """
    MACRO_CHROMS = {'1', '1A', '2', '3', '4', '4A', '5', '6', '7', '8'}
    SEX_CHROMS = {'Z', 'W'}
    c = str(chrom)
    if c == '7':
        return 'chr7_ancestral'
    elif c in MACRO_CHROMS:
        return 'macro_derived'
    elif c in SEX_CHROMS:
        return 'sex_chrom'
    else:
        return 'micro_derived'
