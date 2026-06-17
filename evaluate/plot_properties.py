"""
Property distribution plots: MOSES vs generated molecules from generate.py CSVs.

Usage
-----
python evaluate/plot_properties.py gen_classic_300k.csv
python evaluate/plot_properties.py gen_classic_300k.csv gen_smiles_300k.csv --jobs 8
python evaluate/plot_properties.py gen_classic_300k.csv --moses datasets/molgpt_classic.csv \
                                                         --cache .cache --figure figures
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

MOSES_COLOR  = "#1f77b4"
SAMPLE_COLOR = "#ff7f0e"
KDE_PTS      = 400

DEFAULT_MOSES  = "datasets/molgpt_classic.csv"
DEFAULT_CACHE  = ".cache"
DEFAULT_FIGURE = "figures"

# ── Property computation ──────────────────────────────────────────────────────

def _props_for_smiles(smi: str) -> dict:
    """Compute all 8 properties for a single SMILES string."""
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
    """Compute all properties in parallel with joblib."""
    results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_props_for_smiles)(smi) for smi in smiles_series
    )
    return pd.DataFrame(results, index=smiles_series.index)

# ── Caching ───────────────────────────────────────────────────────────────────

def _file_hash(path: str) -> str:
    """MD5 of the first 64 KB + file size — fast fingerprint."""
    h = hashlib.md5()
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        h.update(f.read(65536))
    h.update(str(size).encode())
    return h.hexdigest()[:16]


def load_moses_props(moses_path: str, cache_dir: str, n_jobs: int = -1) -> pd.DataFrame:
    """
    Load and compute properties for the MOSES reference CSV, using a
    parquet cache keyed on the file's content hash to avoid recomputation.
    """
    os.makedirs(cache_dir, exist_ok=True)
    tag        = _file_hash(moses_path)
    cache_file = os.path.join(cache_dir, f"moses_props_{tag}.parquet")

    if os.path.exists(cache_file):
        print(f"[cache] Loading MOSES properties from {cache_file}")
        return pd.read_parquet(cache_file)

    print(f"[cache] Computing MOSES properties (n_jobs={n_jobs}) ...")
    df    = pd.read_csv(moses_path)
    smiles_col = "smiles" if "smiles" in df.columns else df.columns[0]
    smiles = df[smiles_col].dropna().reset_index(drop=True)
    props  = compute_properties(smiles, n_jobs=n_jobs)
    props.insert(0, "smiles", smiles.values)
    props.to_parquet(cache_file, index=False)
    print(f"[cache] Saved to {cache_file}")
    return props

# ── Generated CSV loader ──────────────────────────────────────────────────────

def load_generated_csv(path: str, n_jobs: int = -1) -> pd.DataFrame:
    """
    Load a CSV from generate.py.  Pre-computed columns (qed, sas, logp, tpsa)
    are reused; the four missing ones are computed in parallel.

    generate.py columns : molecule, smiles, qed, sas, logp, tpsa, ...
    Output columns      : smiles, qed, sa_score, crippen_logp, tpsa,
                          mol_weight, hbd, hba, rot_bonds
    """
    df = pd.read_csv(path, usecols=lambda c: c != "molecule")
    df = df.rename(columns={"sas": "sa_score", "logp": "crippen_logp"})
    df = df.dropna(subset=["smiles"]).reset_index(drop=True)

    missing = [c for c in ["mol_weight", "hbd", "hba", "rot_bonds"] if c not in df.columns]
    if missing:
        print(f"[props] Computing {missing} for {len(df):,} molecules (n_jobs={n_jobs}) ...")
        extra = pd.DataFrame(compute_properties(df["smiles"], n_jobs=n_jobs)[missing])
        df = pd.concat([df, extra], axis=1)

    return df

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


def _count_plot(ax, values, color, bins):
    v      = values[np.isfinite(values)].astype(int)
    counts = np.bincount(v - bins[0], minlength=len(bins))
    freq   = counts / counts.sum()
    width  = 0.38
    offsets = (np.array(bins) - width / 2
               if color == MOSES_COLOR
               else np.array(bins) + width / 2)
    ax.bar(offsets, freq[:len(bins)], width=width,
           color=color, alpha=0.75, edgecolor="white", linewidth=0.4)

# ── Main figure ───────────────────────────────────────────────────────────────

def plot_property_panel(
    moses_props: pd.DataFrame,
    sample_props: pd.DataFrame,
    title: str = "",
    figsize: tuple = (15, 9),
    save_path: str | None = None,
) -> plt.Figure:
    fig, axes = plt.subplots(3, 3, figsize=figsize)
    fig.patch.set_facecolor("white")
    if title:
        fig.suptitle(title, fontsize=12, y=1.01)

    for idx, (key, label) in enumerate(CONTINUOUS + DISCRETE):
        ax = axes[idx // 3][idx % 3]
        ax.set_facecolor("white")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)

        m = moses_props[key].dropna().values  if key in moses_props.columns  else np.array([])
        s = sample_props[key].dropna().values if key in sample_props.columns else np.array([])

        if len(m) == 0 and len(s) == 0:
            ax.set_visible(False)
            continue

        is_discrete = key in {k for k, _ in DISCRETE}

        if is_discrete:
            all_vals = np.concatenate([m, s]).astype(int)
            lo, hi   = int(all_vals.min()), int(all_vals.max())
            bins     = list(range(lo, min(hi + 1, lo + 20)))
            _count_plot(ax, m, MOSES_COLOR,  bins)
            _count_plot(ax, s, SAMPLE_COLOR, bins)
            ax.set_xlim(lo - 1, bins[-1] + 1)
            ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            ax.set_ylabel("Frequency", fontsize=9)
        else:
            all_vals = np.concatenate([m, s])
            finite   = all_vals[np.isfinite(all_vals)]
            lo = np.percentile(finite, 0.5)
            hi = np.percentile(finite, 99.5)
            _kde_plot(ax, m, MOSES_COLOR,  lo, hi)
            _kde_plot(ax, s, SAMPLE_COLOR, lo, hi)
            pad = (hi - lo) * 0.04
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(bottom=0)
            ax.set_ylabel("Density", fontsize=9)

        ax.set_title(label, fontsize=10, pad=5)

    axes[2][2].set_visible(False)
    leg_ax = fig.add_axes([0.68, 0.06, 0.22, 0.20])
    leg_ax.set_axis_off()
    handles = [
        Line2D([0], [0], color=MOSES_COLOR,  linewidth=3, label="MOSES"),
        Line2D([0], [0], color=SAMPLE_COLOR, linewidth=3, label="Generated molecules"),
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
        description="Plot property distributions for generate.py output CSVs vs MOSES."
    )
    parser.add_argument(
        "samples", nargs="+",
        help="One or more generate.py CSV files to plot.",
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
    args = parser.parse_args()

    os.makedirs(args.figure, exist_ok=True)

    moses_props = load_moses_props(args.moses, args.cache, n_jobs=args.jobs)

    for sample_path in args.samples:
        if not os.path.exists(sample_path):
            print(f"[warn] File not found, skipping: {sample_path}", file=sys.stderr)
            continue

        stem       = os.path.splitext(os.path.basename(sample_path))[0]
        save_path  = os.path.join(args.figure, f"{stem}_props.png")

        print(f"\n[plot] {sample_path} → {save_path}")
        sample_props = load_generated_csv(sample_path, n_jobs=args.jobs)
        plot_property_panel(
            moses_props, sample_props,
            title=stem,
            save_path=save_path,
        )
        plt.close("all")


if __name__ == "__main__":
    main()
