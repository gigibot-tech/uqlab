"""
Single place to read outputs from one ``run_fast_uncertainty_classification`` run.

Every run folder should contain (when successful):

- ``summary.json`` — eval sizes, one-vs-rest AUROC, config
- ``per_sample_signals.csv`` — one row per eval sample (group, dataset_index, labels, signals)
- ``results.pt`` — full tensors (optional duplicate of the same evaluation)

Use :func:`load_run_directory` to inspect a run; use :func:`metrics_row_from_run`
when building ``metrics.csv`` rows.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

GROUP_CLEAN, GROUP_ALEATORIC, GROUP_EPISTEMIC = 0, 1, 2

# Columns in ``per_sample_signals.csv`` / ``build_fast_pilot_signal_table``.
FAST_PILOT_SIGNAL_NAMES: tuple[str, ...] = (
    "msp_uncertainty",
    "predictive_entropy",
    "mutual_info",
    "coherence",
    "inverse_coherence",
    "dominance",
    "inverse_mass",
    "inverse_logit_magnitude",
)
_EVAL_PACK_TAGS = ("epistemic", "aleatoric", "clean")


@dataclass(frozen=True)
class RunArtifacts:
    """Normalized view of one experiment output directory."""

    run_dir: Path
    summary_path: Path | None
    per_sample_path: Path | None
    results_pt_path: Path | None
    eval_sizes: dict[str, int] = field(default_factory=dict)
    one_vs_rest_auroc: list[dict[str, Any]] = field(default_factory=list)
    train_size: int | None = None
    source: str = "none"  # "summary.json" | "results.pt" | "none"

    @property
    def has_data(self) -> bool:
        return self.source != "none"

    def auroc_by_signal(self) -> dict[str, dict[str, float]]:
        """``{signal: {"aleatoric": float, "epistemic": float}}``."""
        out: dict[str, dict[str, float]] = {}
        for row in self.one_vs_rest_auroc:
            name = row.get("signal")
            if not name:
                continue
            out[str(name)] = {
                "aleatoric": float(row.get("aleatoric_like_auroc", row.get("aleatoric_auroc", 0))),
                "epistemic": float(row.get("epistemic_like_auroc", row.get("epistemic_auroc", 0))),
            }
        return out


def load_run_directory(run_dir: Path) -> RunArtifacts:
    """
    Load run artifacts from *run_dir*.

    Prefers ``summary.json`` (human-readable, same numbers as the backend).
    Falls back to ``results.pt`` when only that exists.
    """
    run_dir = Path(run_dir)
    summary_path = run_dir / "summary.json"
    per_sample_path = run_dir / "per_sample_signals.csv"
    results_pt_path = run_dir / "results.pt"

    if summary_path.is_file():
        with summary_path.open() as f:
            summary = json.load(f)
        return RunArtifacts(
            run_dir=run_dir,
            summary_path=summary_path,
            per_sample_path=per_sample_path if per_sample_path.is_file() else None,
            results_pt_path=results_pt_path if results_pt_path.is_file() else None,
            eval_sizes=dict(summary.get("eval_sizes") or {}),
            one_vs_rest_auroc=list(summary.get("one_vs_rest_auroc") or []),
            train_size=summary.get("train_size"),
            source="summary.json",
        )

    if results_pt_path.is_file():
        return _artifacts_from_results_pt(
            run_dir, results_pt_path, per_sample_path, summary_path
        )

    return RunArtifacts(
        run_dir=run_dir,
        summary_path=None,
        per_sample_path=per_sample_path if per_sample_path.is_file() else None,
        results_pt_path=None,
        source="none",
    )


def _artifacts_from_results_pt(
    run_dir: Path,
    results_pt_path: Path,
    per_sample_path: Path,
    summary_path: Path,
) -> RunArtifacts:
    import torch

    data = torch.load(results_pt_path, map_location="cpu", weights_only=False)
    one_vs_rest: list[dict[str, Any]] = []
    if "auroc_rows" in data:
        for signal_name, alea_auc, epis_auc in data["auroc_rows"]:
            alea_val = float(alea_auc.item() if hasattr(alea_auc, "item") else alea_auc)
            epis_val = float(epis_auc.item() if hasattr(epis_auc, "item") else epis_auc)
            one_vs_rest.append(
                {
                    "signal": signal_name,
                    "aleatoric_like_auroc": alea_val,
                    "epistemic_like_auroc": epis_val,
                }
            )

    eval_sizes: dict[str, int] = {}
    if "eval_group_labels" in data:
        import numpy as np

        labels = data["eval_group_labels"]
        if hasattr(labels, "cpu"):
            labels = labels.cpu().numpy()
        labels = np.asarray(labels)
        eval_sizes = {
            "clean": int((labels == GROUP_CLEAN).sum()),
            "aleatoric_like": int((labels == GROUP_ALEATORIC).sum()),
            "epistemic_like": int((labels == GROUP_EPISTEMIC).sum()),
        }

    return RunArtifacts(
        run_dir=run_dir,
        summary_path=summary_path if summary_path.is_file() else None,
        per_sample_path=per_sample_path if per_sample_path.is_file() else None,
        results_pt_path=results_pt_path,
        eval_sizes=eval_sizes,
        one_vs_rest_auroc=one_vs_rest,
        train_size=None,
        source="results.pt",
    )


def load_per_sample_table(run_dir: Path, *, max_rows: int = 500) -> pd.DataFrame | None:
    """Load ``per_sample_signals.csv`` if present."""
    path = Path(run_dir) / "per_sample_signals.csv"
    if not path.is_file():
        return None
    df = pd.read_csv(path)
    if len(df) > max_rows:
        return df.head(max_rows)
    return df


def metrics_row_from_run(run_dir: Path) -> dict[str, Any]:
    """
    Build a flat metrics dict for one run (for ``metrics.csv`` / unified loaders).

    Merges AUROC from artifacts with per-signal means from ``results.pt`` when
    available (summary alone does not carry ``<signal>_mean_*`` columns).
    """
    artifacts = load_run_directory(run_dir)
    metrics: dict[str, Any] = {}

    for signal, scores in artifacts.auroc_by_signal().items():
        metrics[f"{signal}_aleatoric_auroc"] = scores["aleatoric"]
        metrics[f"{signal}_epistemic_auroc"] = scores["epistemic"]

    results_pt = run_dir / "results.pt"
    if results_pt.is_file():
        metrics.update(_signal_means_from_results_pt(results_pt))

    if "predictions" in _results_pt_keys(results_pt) and results_pt.is_file():
        import torch

        data = torch.load(results_pt, map_location="cpu", weights_only=False)
        if "predictions" in data and "eval_clean_labels" in data:
            pred = data["predictions"]
            labels = data["eval_clean_labels"]
            if not isinstance(pred, torch.Tensor):
                pred = torch.tensor(pred)
            if not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels)
            metrics["accuracy"] = float((pred == labels).float().mean().item())

    return metrics


def _metric_float(metrics: dict[str, Any], key: str, fallback: str | None = None) -> float | None:
    for k in (key, fallback):
        if not k or k not in metrics or metrics[k] is None:
            continue
        try:
            v = float(metrics[k])
            if v == v:  # not NaN
                return v
        except (TypeError, ValueError):
            continue
    return None


def format_run_metrics_console_lines(metrics: dict[str, Any]) -> list[str]:
    """Human-readable lines for terminal output after one run."""
    lines: list[str] = []

    acc = _metric_float(metrics, "accuracy")
    if acc is not None:
        lines.append(f"   Accuracy: {acc:.4f}")

    lines.append("   --- Signal means (all eval samples) ---")
    n_mean_lines = len(lines)
    for signal in FAST_PILOT_SIGNAL_NAMES:
        val = _metric_float(metrics, f"{signal}_mean")
        if val is not None:
            lines.append(f"   {signal}: {val:.4f}")
    if len(lines) == n_mean_lines:
        lines.append("   (no per-signal means — re-run with results.pt)")

    pool_lines: list[str] = []
    for signal in FAST_PILOT_SIGNAL_NAMES:
        parts: list[str] = []
        for tag in _EVAL_PACK_TAGS:
            val = _metric_float(metrics, f"{signal}_mean_{tag}")
            if val is not None:
                parts.append(f"{tag}={val:.4f}")
        if parts:
            pool_lines.append(f"   {signal}: {', '.join(parts)}")
    if pool_lines:
        lines.append("   --- Means by eval pack ---")
        lines.extend(pool_lines)

    auroc_lines: list[str] = []
    for signal in FAST_PILOT_SIGNAL_NAMES:
        alea = _metric_float(metrics, f"{signal}_aleatoric_auroc")
        epis = _metric_float(metrics, f"{signal}_epistemic_auroc")
        if alea is not None or epis is not None:
            a = f"{alea:.4f}" if alea is not None else "—"
            e = f"{epis:.4f}" if epis is not None else "—"
            auroc_lines.append(f"   {signal}: aleatoric_auroc={a}, epistemic_auroc={e}")
    if auroc_lines:
        lines.append("   --- AUROC (one-vs-rest on eval packs) ---")
        lines.extend(auroc_lines)

    if not lines:
        lines.append("   (no metrics available)")
    return lines


def print_run_metrics_summary(metrics: dict[str, Any]) -> None:
    """Print :func:`format_run_metrics_console_lines` to stdout."""
    for line in format_run_metrics_console_lines(metrics):
        print(line)


def _sweep_key_column(sweep_kind: str) -> str:
    if sweep_kind == "label_noise":
        return "noise_percent"
    if sweep_kind == "dataset_size":
        return "under_train_per_class"
    raise ValueError(f"sweep_kind must be 'label_noise' or 'dataset_size', got {sweep_kind!r}")


def save_signal_formula_manifest(run_dir: Path, manifest: dict[str, Any]) -> Path:
    """Write ``signal_formulas.json`` (operands, operators, eval protocol)."""
    path = Path(run_dir) / "signal_formulas.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(manifest, f, indent=2)
    return path


def save_run_metrics_row_csv(run_dir: Path, metrics: dict[str, Any]) -> Path:
    """Write one experiment's metrics as a single-row CSV inside the run folder."""
    path = Path(run_dir) / "metrics_row.csv"
    pd.DataFrame([metrics]).to_csv(path, index=False)
    return path


def append_metrics_row_to_csv(
    metrics: dict[str, Any],
    csv_path: Path,
    *,
    sweep_kind: str,
) -> int:
    """
    Append one experiment row to ``metrics.csv`` immediately (dedupe on write).

    Dedupe key: ``(architecture, dataset_size|noise_percent)``; latest
    ``timestamp`` wins. Returns total row count after write.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    key_col = _sweep_key_column(sweep_kind)
    if key_col not in metrics and sweep_kind == "dataset_size" and "dataset_size" in metrics:
        metrics = {**metrics, key_col: metrics["dataset_size"]}

    row_df = pd.DataFrame([metrics])
    if csv_path.is_file():
        try:
            existing = pd.read_csv(csv_path)
        except Exception:
            existing = pd.DataFrame()
        combined = pd.concat([existing, row_df], ignore_index=True, sort=False)
    else:
        combined = row_df

    if combined.empty:
        combined.to_csv(csv_path, index=False)
        return 0

    if {"architecture", key_col}.issubset(combined.columns):
        if "timestamp" in combined.columns:
            combined = combined.sort_values("timestamp", kind="stable")
        combined = combined.drop_duplicates(
            subset=["architecture", key_col],
            keep="last",
        )
        combined = combined.sort_values(["architecture", key_col]).reset_index(drop=True)

    combined.to_csv(csv_path, index=False)
    return len(combined)


def save_zwischen_result(run_dir: Path, stage: str, payload: dict[str, Any]) -> Path:
    """
    Persist intermediate eval artifacts under ``<run_dir>/zwischen/<stage>.pt``.

    Always written during fast-pilot eval so runs can resume/debug without
    re-running DualXDA or MC dropout.
    """
    import torch

    zwischen = Path(run_dir) / "zwischen"
    zwischen.mkdir(parents=True, exist_ok=True)
    safe = stage.replace(" ", "_").replace("/", "_")
    path = zwischen / f"{safe}.pt"
    torch.save(payload, path)
    manifest = zwischen / "manifest.json"
    entries: list[dict[str, str]] = []
    if manifest.is_file():
        try:
            entries = json.loads(manifest.read_text())
        except json.JSONDecodeError:
            entries = []
    entries = [e for e in entries if e.get("stage") != safe]
    entries.append({"stage": safe, "path": str(path.name)})
    manifest.write_text(json.dumps(entries, indent=2))
    return path


def _results_pt_keys(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    import torch

    data = torch.load(path, map_location="cpu", weights_only=False)
    return set(data.keys()) if isinstance(data, dict) else set()


def _signal_means_from_results_pt(results_pt: Path) -> dict[str, float]:
    """Extract ``<signal>_mean*`` columns (same logic as validation rebuild)."""
    import numpy as np
    import torch

    data = torch.load(results_pt, map_location="cpu", weights_only=False)
    if "signal_table" not in data:
        return {}

    signal_table = data["signal_table"]
    if hasattr(signal_table, "columns"):
        signal_iter = {name: signal_table[name].to_numpy() for name in signal_table.columns}
    elif isinstance(signal_table, dict):
        signal_iter = {}
        for name, values in signal_table.items():
            if hasattr(values, "cpu"):
                values = values.cpu().numpy()
            signal_iter[name] = np.asarray(values)
    else:
        return {}

    group_labels = data.get("eval_group_labels")
    if group_labels is not None and hasattr(group_labels, "cpu"):
        group_labels = group_labels.cpu().numpy()
    elif group_labels is not None:
        group_labels = np.asarray(group_labels)

    metrics: dict[str, float] = {}
    for name, values in signal_iter.items():
        if values is None or len(values) == 0:
            continue
        metrics[f"{name}_mean"] = float(np.nanmean(values))
        if group_labels is not None and group_labels.shape == values.shape:
            for tag, code in (
                ("epistemic", GROUP_EPISTEMIC),
                ("aleatoric", GROUP_ALEATORIC),
                ("clean", GROUP_CLEAN),
            ):
                mask = group_labels == code
                if mask.any():
                    metrics[f"{name}_mean_{tag}"] = float(np.nanmean(values[mask]))

    # Row-1 plot proxies: mean_total_* aliases for notebooks; values are MC-dropout signals.
    # total = predictive_entropy, epistemic = mutual_info, aleatoric = total − epistemic.
    if "predictive_entropy_mean" in metrics:
        metrics["mean_total_uncertainty"] = metrics["predictive_entropy_mean"]
    if "mutual_info_mean" in metrics:
        metrics["mean_epistemic_uncertainty"] = metrics["mutual_info_mean"]
    if "mean_total_uncertainty" in metrics and "mean_epistemic_uncertainty" in metrics:
        metrics["mean_aleatoric_uncertainty"] = (
            metrics["mean_total_uncertainty"] - metrics["mean_epistemic_uncertainty"]
        )

    return metrics
