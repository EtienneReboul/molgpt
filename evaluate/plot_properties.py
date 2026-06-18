"""
Property distribution plots: MOSES vs generated molecules from generate.py CSVs.
All sample CSVs are overlaid on one combined figure.

Usage
-----
# auto-detects dataset type from filename keywords (classic / block / smiles_sel)
python evaluate/plot_properties.py gen_classic_300k.csv gen_block_300k.csv
python evaluate/plot_properties.py gen_classic_300k.csv gen_block_300k.csv gen_smiles_selection_300k.csv --jobs 8
"""

import argparse
import hashlib
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
from joblib import Parallel, delayed
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, rdMolDescriptors

# ── Column definitions ────────────────────────────────────────────────────────

CONTINUOUS = [
    ("mol_weight",   "Molecular Weight (Da)"),
    ("crippen_logp", "Crippen LogP"),
    ("qed",          "QED"),
    ("sa_score",     "SA Score"),
    ("tpsa",         "TPSA (Å²)"),
]
DISCRETE = [
    ("hbd",       "H-Bond Donors"),
    ("hba",       "H-Bond Acceptors"),
    ("rot_bonds", "Rotatable Bonds"),
]

ALL_COMPUTED = ["mol_weight", "crippen_logp", "qed", "sa_score", "tpsa", "hbd", "hba", "rot_bonds"]

KDE_PTS = 400

# ── Dataset identity: keyword → (legend label, color) ────────────────────────
# Matched against the CSV filename (lowercase).

DATASET_STYLES: list[tuple[str, str, str]] = [
    # (filename keyword, legend label, hex color)
    ("classic",    "RDKIT default canonical SMILES", "#ff7f0e"),
    ("block",      "block SMILES",                   "#2ca02c"),
    ("smiles_sel", "selected SMILES",                "#d62728"),
]

MOSES_LABEL = "MOSES"
MOSES_COLOR = "#1f77b4"

DEFAULT_MOSES  = "datasets/molgpt_classic.csv"
DEFAULT_CACHE  = ".cache"
DEFAULT_FIGURE = "figures"

# ── Property computation ──────────────────────────────────────────────────────

def _props_for_smiles(smi: str) -> dict:
    nan_row = {k: np.nan for k in ALL_COMPUTED}
    if not isinstance(smi, str):
        return nan_row
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return nan_row
    return dict(
        mol_weight   = Descriptors.MolWt(mol),
        crippen_logp = Descriptors.MolLogP(mol),
        qed          = QED.qed(mol),
        sa_score     = _sa_score(mol),
        tpsa         = rdMolDescriptors.CalcTPSA(mol),
        hbd          = rdMolDescriptors.CalcNumHBD(mol),
        hba          = rdMolDescriptors.CalcNumHBA(mol),
        rot_bonds    = rdMolDescriptors.CalcNumRotatableBonds(mol),
    )


_sascorer = None  # cached per-process (one load per joblib worker)


def _get_sascorer():
    global _sascorer
    if _sascorer is None:
        import importlib.util
        import pathlib
        from rdkit.Chem import RDConfig
        sa_path = pathlib.Path(RDConfig.RDContribDir) / "SA_Score" / "sascorer.py"
        spec = importlib.util.spec_from_file_location("sascorer", sa_path)
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        _sascorer = mod
    return _sascorer


def _sa_score(mol) -> float:
    try:
        scorer = _get_sascorer()
        if scorer is None:
            return np.nan
        return float(scorer.calculateScore(mol))
    except Exception:
        return np.nan


def compute_properties(smiles_series: pd.Series, n_jobs: int = -1) -> pd.DataFrame:
    """Compute all properties in parallel. Uses fork-based multiprocessing — sascorer is
    mostly pure Python (GIL-bound), so threads don't help; loky workers crash with RDKit."""
    results = Parallel(n_jobs=n_jobs, backend="multiprocessing", verbose=0)(
        delayed(_props_for_smiles)(smi) for smi in smiles_series
    )
    return pd.DataFrame(results, index=smiles_series.index)

# ── Caching ───────────────────────────────────────────────────────────────────

def _file_hash(path: str) -> str:
    h = hashlib.md5()
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        h.update(f.read(65536))
    h.update(str(size).encode())
    return h.hexdigest()[:16]


def load_moses_props(moses_path: str, cache_dir: str, n_jobs: int = -1) -> pd.DataFrame:
    """Load MOSES properties from parquet cache, computing them if needed."""
    os.makedirs(cache_dir, exist_ok=True)
    tag        = _file_hash(moses_path)
    cache_file = os.path.join(cache_dir, f"moses_props_{tag}.parquet")

    if os.path.exists(cache_file):
        print(f"[cache] Loading MOSES properties from {cache_file}")
        return pd.read_parquet(cache_file)

    print(f"[cache] Computing MOSES properties (n_jobs={n_jobs}) ...")
    df         = pd.read_csv(moses_path)
    smiles_col = "smiles" if "smiles" in df.columns else df.columns[0]
    smiles     = df[smiles_col].dropna().reset_index(drop=True)
    props      = compute_properties(smiles, n_jobs=n_jobs)
    props.insert(0, "smiles", smiles.values)
    props.to_parquet(cache_file, index=False)
    print(f"[cache] Saved to {cache_file}")
    return props

# ── Generated CSV loader ──────────────────────────────────────────────────────

def load_generated_csv(path: str, n_jobs: int = -1) -> pd.DataFrame:
    """Load a generate.py CSV, rename columns, compute missing properties."""
    df = pd.read_csv(path, usecols=lambda c: c != "molecule")
    df = df.rename(columns={"sas": "sa_score", "logp": "crippen_logp"})
    df = df.dropna(subset=["smiles"]).reset_index(drop=True)

    missing = [c for c in ["mol_weight", "hbd", "hba", "rot_bonds"] if c not in df.columns]
    if missing:
        print(f"[props] Computing {missing} for {len(df):,} molecules (n_jobs={n_jobs}) ...")
        extra = pd.DataFrame(compute_properties(df["smiles"], n_jobs=n_jobs)[missing])
        df    = pd.concat([df, extra], axis=1)

    return df


def label_and_color(csv_path: str) -> tuple[str, str]:
    """Infer legend label and color from the CSV filename."""
    name = os.path.basename(csv_path).lower()
    for keyword, label, color in DATASET_STYLES:
        if keyword in name:
            return label, color
    return os.path.splitext(os.path.basename(csv_path))[0], "#9467bd"


def build_moses_smiles_set(moses_path: str) -> set[str]:
    """Return the set of SMILES from the MOSES CSV (assumed already canonical)."""
    df         = pd.read_csv(moses_path)
    smiles_col = "smiles" if "smiles" in df.columns else df.columns[0]
    return set(df[smiles_col].dropna())


def _canonicalize(smi: str) -> str | None:
    if not isinstance(smi, str):
        return None
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol) if mol is not None else None


def filter_novel_unique(df: pd.DataFrame, moses_smiles_set: set[str]) -> pd.DataFrame:
    """
    Keep only novel (not in MOSES) and unique molecules.
    Canonicalizes generated SMILES before both comparisons.
    """
    n0 = len(df)
    df = df.copy()
    df["_canon"] = df["smiles"].apply(_canonicalize)

    df = df.dropna(subset=["_canon"])
    n_valid = len(df)

    df = df.drop_duplicates(subset=["_canon"])
    n_unique = len(df)

    df = df[~df["_canon"].isin(moses_smiles_set)]
    n_novel = len(df)

    print(
        f"[filter] {n0:,} total  →  {n_valid:,} valid"
        f"  →  {n_unique:,} unique  →  {n_novel:,} novel"
        f"  (novelty {n_novel/n_unique:.1%}, uniqueness {n_unique/n_valid:.1%})"
    )
    return df.drop(columns=["_canon"]).reset_index(drop=True)

# ── Plot helpers ──────────────────────────────────────────────────────────────

def _kde_plot(ax, values, color, lo, hi):
    v = values[np.isfinite(values)]
    if len(v) < 10:
        return
    pad = (hi - lo) * 0.04
    x   = np.linspace(lo - pad, hi + pad, KDE_PTS)
    y   = gaussian_kde(v, bw_method="scott")(x)
    ax.plot(x, y, color=color, linewidth=2.0)
    ax.fill_between(x, y, alpha=0.15, color=color)


def _count_plot(ax, values, color, bins, n_datasets: int, dataset_idx: int):
    """Side-by-side bars, offset so all datasets fit without overlap."""
    v      = values[np.isfinite(values)].astype(int)
    counts = np.bincount(v - bins[0], minlength=len(bins))
    freq   = counts / counts.sum()
    total_width = 0.7
    width   = total_width / n_datasets
    shift   = (dataset_idx - (n_datasets - 1) / 2) * width
    offsets = np.array(bins) + shift
    ax.bar(offsets, freq[:len(bins)], width=width * 0.9,
           color=color, alpha=0.75, edgecolor="white", linewidth=0.4)

# ── Main figure ───────────────────────────────────────────────────────────────

def plot_property_panel(
    datasets: list[tuple[str, str, pd.DataFrame]],
    figsize: tuple = (15, 9),
    save_path: str | None = None,
) -> plt.Figure:
    """
    Combined 3×3 panel overlaying all datasets.

    Parameters
    ----------
    datasets : list of (label, color, DataFrame) — MOSES first, then generated.
    """
    fig, axes = plt.subplots(3, 3, figsize=figsize)
    fig.patch.set_facecolor("white")

    n_datasets = len(datasets)

    for idx, (key, panel_title) in enumerate(CONTINUOUS + DISCRETE):
        ax = axes[idx // 3][idx % 3]
        ax.set_facecolor("white")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)

        arrays = []
        for _label, _color, df in datasets:
            arr = df[key].dropna().values if key in df.columns else np.array([])
            arrays.append(arr)

        if all(len(a) == 0 for a in arrays):
            ax.set_visible(False)
            continue

        is_discrete = key in {k for k, _ in DISCRETE}
        all_vals    = np.concatenate([a for a in arrays if len(a) > 0])

        if is_discrete:
            all_int  = all_vals.astype(int)
            lo, hi   = int(all_int.min()), int(all_int.max())
            bins     = list(range(lo, min(hi + 1, lo + 20)))
            for i, ((_label, color, _df), arr) in enumerate(zip(datasets, arrays)):
                if len(arr) > 0:
                    _count_plot(ax, arr, color, bins, n_datasets, i)
            ax.set_xlim(lo - 1, bins[-1] + 1)
            ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            ax.set_ylabel("Frequency", fontsize=9)
        else:
            finite = all_vals[np.isfinite(all_vals)]
            lo     = np.percentile(finite, 0.5)
            hi     = np.percentile(finite, 99.5)
            for (_label, color, _df), arr in zip(datasets, arrays):
                if len(arr) > 0:
                    _kde_plot(ax, arr, color, lo, hi)
            pad = (hi - lo) * 0.04
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(bottom=0)
            ax.set_ylabel("Density", fontsize=9)

        ax.set_title(panel_title, fontsize=10, pad=5)

    # ── Legend in the 9th cell ────────────────────────────────────────────────
    axes[2][2].set_visible(False)
    leg_ax = fig.add_axes([0.68, 0.06, 0.28, 0.26])
    leg_ax.set_axis_off()
    handles = [
        Line2D([0], [0], color=color, linewidth=3, label=label)
        for label, color, _ in datasets
    ]
    leg = leg_ax.legend(
        handles=handles, title="Dataset",
        title_fontsize=10, fontsize=9,
        loc="center", frameon=True,
        framealpha=0.85, edgecolor="#cccccc",
    )
    leg.get_frame().set_linewidth(0.6)

    fig.tight_layout(rect=[0, 0, 1, 1])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Saved: {save_path}")

    return fig

# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot combined property distributions for generate.py CSVs vs MOSES."
    )
    parser.add_argument(
        "samples", nargs="+",
        help="One or more generate.py CSV files (classic / block / smiles_sel).",
    )
    parser.add_argument(
        "--moses", default=DEFAULT_MOSES,
        help=f"MOSES reference CSV (default: {DEFAULT_MOSES}).",
    )
    parser.add_argument(
        "--cache", default=DEFAULT_CACHE,
        help=f"Directory for cached MOSES properties (default: {DEFAULT_CACHE}).",
    )
    parser.add_argument(
        "--figure", default=DEFAULT_FIGURE,
        help=f"Directory to save figures (default: {DEFAULT_FIGURE}).",
    )
    parser.add_argument(
        "--jobs", type=int, default=-1,
        help="Number of parallel workers for joblib (default: -1 = all CPUs).",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output filename (stem only, no extension). Default: combined names.",
    )
    args = parser.parse_args()

    os.makedirs(args.figure, exist_ok=True)

    moses_props      = load_moses_props(args.moses, args.cache, n_jobs=args.jobs)
    moses_smiles_set = build_moses_smiles_set(args.moses)
    datasets: list[tuple[str, str, pd.DataFrame]] = [(MOSES_LABEL, MOSES_COLOR, moses_props)]

    for sample_path in args.samples:
        if not os.path.exists(sample_path):
            print(f"[warn] File not found, skipping: {sample_path}", file=sys.stderr)
            continue
        label, color = label_and_color(sample_path)
        print(f"\n[load] {sample_path}  →  \"{label}\"")
        df = load_generated_csv(sample_path, n_jobs=args.jobs)
        df = filter_novel_unique(df, moses_smiles_set)
        datasets.append((label, color, df))

    if len(datasets) == 1:
        print("[error] No valid sample files found.", file=sys.stderr)
        sys.exit(1)

    stem      = args.out or "_vs_".join(
        os.path.splitext(os.path.basename(p))[0] for p in args.samples
    )
    save_path = os.path.join(args.figure, f"{stem}_props.png")

    print(f"\n[plot] → {save_path}")
    plot_property_panel(datasets, save_path=save_path)
    plt.close("all")


if __name__ == "__main__":
    main()
