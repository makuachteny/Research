"""
Species configuration loader for the MROH multicopy analysis pipeline.

Each species has a JSON config file in configs/ that defines:
  - Input file paths (tBLASTn, exon 13 anchors, reference proteins)
  - Gene structure (exon boundaries, analysis range)
  - Chromosome classification (ancestral, macro, micro, sex)
  - Analysis parameters (coverage threshold, PAML settings)

Usage:
    from species_config import load_config
    cfg = load_config("melospiza_georgiana")
    # or
    cfg = load_config("configs/melospiza_georgiana.json")
"""
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"


def load_config(species_or_path):
    """Load species configuration from a JSON file.

    Args:
        species_or_path: Either a species slug (e.g., "melospiza_georgiana")
            or a path to a JSON config file.

    Returns:
        dict with all configuration parameters, paths resolved to absolute.
    """
    path = Path(species_or_path)
    if not path.suffix:
        path = CONFIGS_DIR / f"{species_or_path}.json"
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    with open(path) as f:
        cfg = json.load(f)

    cfg["_config_path"] = str(path)
    cfg["_project_root"] = str(PROJECT_ROOT)

    # Derive species slug from filename
    cfg["species_slug"] = Path(path).stem

    # Compute exon range parameters
    exon_range = cfg["analysis_exon_range"]
    boundaries = cfg["exon_boundaries_aa"]
    start_exon, end_exon = str(exon_range[0]), str(exon_range[1])
    start_aa = boundaries[start_exon][0]
    end_aa = boundaries[end_exon][1]
    cfg["analysis_start_aa"] = start_aa
    cfg["analysis_end_aa"] = end_aa
    cfg["analysis_len_aa"] = end_aa - start_aa + 1
    cfg["analysis_start_nt"] = (start_aa - 1) * 3
    cfg["analysis_end_nt"] = end_aa * 3
    cfg["analysis_len_nt"] = cfg["analysis_len_aa"] * 3

    # Compute fallback exon range if present
    if cfg.get("fallback_exon_range"):
        fb = cfg["fallback_exon_range"]
        fb_start_exon, fb_end_exon = str(fb[0]), str(fb[1])
        fb_start_aa = boundaries[fb_start_exon][0]
        fb_end_aa = boundaries[fb_end_exon][1]
        cfg["fallback_start_aa"] = fb_start_aa
        cfg["fallback_end_aa"] = fb_end_aa
        cfg["fallback_len_aa"] = fb_end_aa - fb_start_aa + 1
        cfg["fallback_len_nt"] = cfg["fallback_len_aa"] * 3

    # Build output prefix from gene name and species slug
    cfg["output_prefix"] = f"{cfg['gene_name'].lower()}_{cfg['species_slug']}"

    return cfg


def resolve_path(cfg, relative_path):
    """Resolve a path relative to the project root."""
    p = Path(relative_path)
    if p.is_absolute():
        return p
    return Path(cfg["_project_root"]) / p


def get_data_dirs(cfg):
    """Return standard data/results directories, creating them if needed."""
    root = Path(cfg["_project_root"])
    dirs = {
        "data_raw": root / "data" / "raw",
        "data_proc": root / "data" / "processed" / cfg["species_slug"],
        "paml_dir": root / "data" / "processed" / cfg["species_slug"] / "paml_input",
        "fig_dir": root / "results" / cfg["species_slug"] / "figures",
        "table_dir": root / "results" / cfg["species_slug"] / "tables",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def classify_chrom(chrom, cfg):
    """Classify a chromosome based on species config."""
    c = str(chrom).rstrip(",")
    if c == cfg["ancestral_chromosome"]:
        return f"chr{c}_ancestral"
    elif c in cfg.get("macro_chromosomes", []):
        return "macro_derived"
    elif c in cfg.get("sex_chromosomes", []):
        return "sex_chrom"
    else:
        return "micro_derived"


def list_available_configs():
    """List all available species config files."""
    configs = sorted(CONFIGS_DIR.glob("*.json"))
    return {c.stem: c for c in configs}
