"""
Aspect 7 validation for four-region experiments.

Noise sweep on the noisy region (label flip %) and sparsity sweep on the sparse
region (train fraction %). Correlation report checks:

- Monotonic: aleatoric metrics track noise; epistemic metrics track sparsity.
- Orthogonal: noise sweep does not move epistemic metrics; sparsity sweep does
  not move aleatoric metrics.
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

from scipy import stats

from uqlab.data.class_regions import DEFAULT_FOUR_REGION_PRESET, normalize_class_regions

NOISE_SWEEP_PCTS: tuple[int, ...] = (0, 10, 25, 50, 75, 100)
SPARSITY_SWEEP_PCTS: tuple[int, ...] = (1, 5, 10, 25, 50, 100)

DEFAULT_ALEATORIC_METRICS: tuple[str, ...] = (
    "inverse_coherence_dualxda",
    "inverse_coherence_graddot",
    "expected_entropy",
)
DEFAULT_EPISTEMIC_METRICS: tuple[str, ...] = (
    "inverse_mass_dualxda",
    "inverse_dominance_dualxda",
    "inverse_mass_graddot",
    "inverse_dominance_graddot",
    "mutual_info",
)


def noise_sweep_regions(
    base: Mapping[str, Any] | None = None,
    pcts: Sequence[int] = NOISE_SWEEP_PCTS,
) -> list[tuple[int, dict[str, dict[str, Any]]]]:
    """Return (flip_pct, class_regions) pairs for noisy-region label-flip sweeps."""
    base_norm = normalize_class_regions(base or DEFAULT_FOUR_REGION_PRESET)
    out: list[tuple[int, dict[str, dict[str, Any]]]] = []
    for pct in pcts:
        regions = copy.deepcopy(base_norm)
        regions["noisy"]["label_flip_pct"] = float(pct)
        out.append((int(pct), regions))
    return out


def sparsity_sweep_regions(
    base: Mapping[str, Any] | None = None,
    pcts: Sequence[int] = SPARSITY_SWEEP_PCTS,
) -> list[tuple[int, dict[str, dict[str, Any]]]]:
    """Return (train_fraction_pct, class_regions) pairs for sparse-region sweeps."""
    base_norm = normalize_class_regions(base or DEFAULT_FOUR_REGION_PRESET)
    out: list[tuple[int, dict[str, dict[str, Any]]]] = []
    for pct in pcts:
        regions = copy.deepcopy(base_norm)
        regions["sparse"]["train_fraction"] = float(pct) / 100.0
        out.append((int(pct), regions))
    return out


def _spearman(x: Sequence[float | int | None], y: Sequence[float | int | None]) -> tuple[float, float]:
    pairs = [
        (float(a), float(b))
        for a, b in zip(x, y)
        if a is not None and b is not None and a == a and b == b
    ]
    if len(pairs) < 3:
        return float("nan"), float("nan")
    xs, ys = zip(*pairs)
    result = stats.spearmanr(xs, ys)
    return float(result.correlation), float(result.pvalue)


@dataclass(frozen=True)
class CorrelationRow:
    sweep_kind: str
    metric: str
    check: str
    spearman_r: float
    p_value: float
    passed: bool


@dataclass
class FourRegionValidationReport:
    noise_rows: list[dict[str, Any]]
    sparsity_rows: list[dict[str, Any]]
    correlations: list[CorrelationRow]
    monotonic_passed: bool
    orthogonal_passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "noise_rows": self.noise_rows,
            "sparsity_rows": self.sparsity_rows,
            "correlations": [asdict(row) for row in self.correlations],
            "monotonic_passed": self.monotonic_passed,
            "orthogonal_passed": self.orthogonal_passed,
            "passed": self.monotonic_passed and self.orthogonal_passed,
        }


def build_correlation_report(
    noise_rows: Sequence[Mapping[str, Any]],
    sparsity_rows: Sequence[Mapping[str, Any]],
    *,
    aleatoric_metrics: Iterable[str] = DEFAULT_ALEATORIC_METRICS,
    epistemic_metrics: Iterable[str] = DEFAULT_EPISTEMIC_METRICS,
    monotonic_threshold: float = 0.5,
    orthogonal_threshold: float = 0.3,
) -> FourRegionValidationReport:
    """Summarize monotonic and orthogonal checks across sweep metric rows."""
    alea = tuple(aleatoric_metrics)
    epi = tuple(epistemic_metrics)
    correlations: list[CorrelationRow] = []

    noise_x = [row.get("noise_pct") for row in noise_rows]
    sparse_x = [row.get("sparse_train_pct") for row in sparsity_rows]

    for metric in alea:
        r, p = _spearman(noise_x, [row.get(metric) for row in noise_rows])
        passed = r == r and r >= monotonic_threshold
        correlations.append(
            CorrelationRow("noise", metric, "monotonic", r, p, passed)
        )

    for metric in epi:
        r, p = _spearman(noise_x, [row.get(metric) for row in noise_rows])
        passed = (r != r) or abs(r) < orthogonal_threshold
        correlations.append(
            CorrelationRow("noise", metric, "orthogonal", r, p, passed)
        )

    for metric in epi:
        r, p = _spearman(sparse_x, [row.get(metric) for row in sparsity_rows])
        passed = r == r and r <= -monotonic_threshold
        correlations.append(
            CorrelationRow("sparsity", metric, "monotonic", r, p, passed)
        )

    for metric in alea:
        r, p = _spearman(sparse_x, [row.get(metric) for row in sparsity_rows])
        passed = (r != r) or abs(r) < orthogonal_threshold
        correlations.append(
            CorrelationRow("sparsity", metric, "orthogonal", r, p, passed)
        )

    monotonic_passed = all(row.passed for row in correlations if row.check == "monotonic")
    orthogonal_passed = all(row.passed for row in correlations if row.check == "orthogonal")
    return FourRegionValidationReport(
        noise_rows=list(noise_rows),
        sparsity_rows=list(sparsity_rows),
        correlations=correlations,
        monotonic_passed=monotonic_passed,
        orthogonal_passed=orthogonal_passed,
    )


def mock_sweep_metric_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Idealized sweep rows for tests (strong monotonic + orthogonal patterns)."""
    noise_rows: list[dict[str, Any]] = []
    for pct in NOISE_SWEEP_PCTS:
        noise_rows.append(
            {
                "noise_pct": pct,
                "inverse_coherence_dualxda": 0.1 + 0.008 * pct,
                "inverse_coherence_graddot": 0.12 + 0.007 * pct,
                "expected_entropy": 0.05 + 0.009 * pct,
                "inverse_mass_dualxda": 0.5,
                "inverse_dominance_dualxda": 0.4,
                "inverse_mass_graddot": 0.48,
                "inverse_dominance_graddot": 0.42,
                "mutual_info": 0.3,
            }
        )

    sparsity_rows: list[dict[str, Any]] = []
    for pct in SPARSITY_SWEEP_PCTS:
        sparsity_rows.append(
            {
                "sparse_train_pct": pct,
                "inverse_mass_dualxda": 0.9 - 0.008 * pct,
                "inverse_dominance_dualxda": 0.85 - 0.007 * pct,
                "inverse_mass_graddot": 0.88 - 0.0075 * pct,
                "inverse_dominance_graddot": 0.82 - 0.006 * pct,
                "mutual_info": 0.7 - 0.005 * pct,
                "inverse_coherence_dualxda": 0.35,
                "inverse_coherence_graddot": 0.33,
                "expected_entropy": 0.25,
            }
        )
    return noise_rows, sparsity_rows


def report_to_json(report: FourRegionValidationReport, *, indent: int = 2) -> str:
    return json.dumps(report.to_dict(), indent=indent)


def _metric_scalar(
    metrics: Mapping[str, Any],
    signal: str,
    *,
    pool: str | None = None,
) -> float | None:
    """Resolve one scalar for correlation rows from a flat metrics dict."""
    candidates: list[str] = []
    if pool:
        candidates.extend(
            [
                f"{signal}_mean_{pool}",
                f"{signal}_{pool}_auroc",
            ]
        )
    candidates.extend([f"{signal}_mean", signal])
    for key in candidates:
        val = metrics.get(key)
        if val is None:
            continue
        try:
            fval = float(val)
            if fval == fval:
                return fval
        except (TypeError, ValueError):
            continue
    return None


def extract_metric_row_from_run(
    metrics: Mapping[str, Any],
    *,
    noise_pct: int | None = None,
    sparse_train_pct: int | None = None,
    aleatoric_metrics: Iterable[str] = DEFAULT_ALEATORIC_METRICS,
    epistemic_metrics: Iterable[str] = DEFAULT_EPISTEMIC_METRICS,
) -> dict[str, Any]:
    """Build one correlation-report row from :func:`metrics_row_from_run` output."""
    row: dict[str, Any] = {}
    if noise_pct is not None:
        row["noise_pct"] = int(noise_pct)
    if sparse_train_pct is not None:
        row["sparse_train_pct"] = int(sparse_train_pct)
    for metric in aleatoric_metrics:
        val = _metric_scalar(metrics, metric, pool="aleatoric")
        if val is not None:
            row[metric] = val
    for metric in epistemic_metrics:
        val = _metric_scalar(metrics, metric, pool="epistemic")
        if val is not None:
            row[metric] = val
    return row


def build_four_region_experiment_config(
    class_regions: Mapping[str, Any],
    *,
    mode: str = "quick",
    dataset: str = "fashion_mnist",
    architecture: str = "pixel_mlp",
) -> dict[str, Any]:
    """YAML-shaped config for a single four-region validation run."""
    from uqlab_orchestrator.config.validation_config import (
        DEFAULT_DATA,
        DEFAULT_PATHS,
        TRAINING_CONFIG,
    )

    train = TRAINING_CONFIG[mode]
    if architecture == "pixel_mlp":
        model_block = {
            "architecture": "pixel_mlp",
            "training_mode": "end_to_end",
            "hidden_dim": 256,
            "dropout": 0.1,
        }
    else:
        model_block = {
            "architecture": "dinov2_mlp",
            "training_mode": "feature_space",
            "hidden_dim": 256,
            "dropout": 0.1,
            "dinov2_model": "small",
        }

    return {
        "seed": 42,
        "device": "auto",
        "data": {
            **DEFAULT_DATA,
            "dataset": dataset,
            "partition_mode": "four_region",
            "class_regions": normalize_class_regions(class_regions),
            "eval_per_group": 100 if mode == "quick" else 600,
            "regular_train_per_class": 300,
            "under_train_per_class": 50,
            "aleatoric_noise_percentage": 0.0,
        },
        "model": model_block,
        "training": {
            "epochs": min(train["epochs"], 5) if mode == "quick" else train["epochs"],
            "learning_rate": train["learning_rate"],
            "weight_decay": train["weight_decay"],
            "train_batch_size": train["train_batch_size"],
            "feature_batch_size": train["feature_batch_size"],
        },
        "evaluation": {
            "mc_passes": train["mc_passes"],
            "top_k": 10,
        },
        "paths": dict(DEFAULT_PATHS),
        "signals": {
            "attribution_backends": ["dualxda", "graddot"],
        },
    }


def _quick_sweep_pcts(pcts: Sequence[int], mode: str) -> list[int]:
    if mode != "quick":
        return list(pcts)
    if len(pcts) <= 3:
        return list(pcts)
    return [pcts[0], pcts[len(pcts) // 2], pcts[-1]]


def run_four_region_sweep_inprocess(
    sweep_kind: str,
    mode: str = "quick",
    *,
    output_base: Any,
    dataset: str = "fashion_mnist",
    architecture: str = "pixel_mlp",
    on_line: Any = None,
) -> tuple[bool, str]:
    """
    Run noise or sparsity four-region sweeps in-process (Streamlit / CLI).

    ``sweep_kind`` is ``noise`` or ``sparsity``.
    """
    import contextlib
    import io
    import tempfile
    from pathlib import Path

    from uqlab.runner.pipeline import run as pipeline_run
    from uqlab.run_artifacts import metrics_row_from_run

    output_base = Path(output_base)
    output_base.mkdir(parents=True, exist_ok=True)
    captured: list[str] = []

    class _LineWriter(io.TextIOBase):
        def write(self, s: str) -> int:
            for line in s.splitlines():
                if not line.strip():
                    continue
                captured.append(line)
                if on_line is not None:
                    on_line(line)
            return len(s)

    def _emit(msg: str) -> None:
        captured.append(msg)
        if on_line is not None:
            on_line(msg)

    try:
        with contextlib.redirect_stdout(_LineWriter()):
            if sweep_kind == "noise":
                presets = noise_sweep_regions(pcts=_quick_sweep_pcts(NOISE_SWEEP_PCTS, mode))
                subdir = output_base / "noise_sweep"
            elif sweep_kind == "sparsity":
                presets = sparsity_sweep_regions(pcts=_quick_sweep_pcts(SPARSITY_SWEEP_PCTS, mode))
                subdir = output_base / "sparsity_sweep"
            else:
                raise ValueError(f"sweep_kind must be 'noise' or 'sparsity', got {sweep_kind!r}")

            subdir.mkdir(parents=True, exist_ok=True)
            import yaml

            for pct, regions in presets:
                name = f"noise{pct}" if sweep_kind == "noise" else f"sparse{pct}"
                run_dir = subdir / name
                if (run_dir / "results.pt").is_file() or (run_dir / "summary.json").is_file():
                    _emit(f"Skipping existing run {name}")
                    continue
                run_dir.mkdir(parents=True, exist_ok=True)
                config = build_four_region_experiment_config(
                    regions,
                    mode=mode,
                    dataset=dataset,
                    architecture=architecture,
                )
                with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
                    yaml.dump(config, f)
                    config_path = Path(f.name)
                try:
                    _emit(f"Running {name} …")
                    pipeline_run(config_path, run_dir)
                finally:
                    config_path.unlink(missing_ok=True)

                metrics = metrics_row_from_run(run_dir)
                row = extract_metric_row_from_run(
                    metrics,
                    noise_pct=pct if sweep_kind == "noise" else None,
                    sparse_train_pct=pct if sweep_kind == "sparsity" else None,
                )
                (run_dir / "four_region_metrics.json").write_text(json.dumps(row, indent=2))
                _emit(f"Done {name}: {row}")

        return True, "\n".join(captured)
    except Exception as exc:
        _emit(f"Four-region sweep failed: {exc}")
        return False, "\n".join(captured)


def load_four_region_sweep_rows_from_disk(
    root: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load aggregated rows from ``results/validation/four_region/*_sweep/*/four_region_metrics.json``."""
    from pathlib import Path

    from uqlab.run_artifacts import metrics_row_from_run

    root = Path(root)
    noise_rows: list[dict[str, Any]] = []
    sparsity_rows: list[dict[str, Any]] = []

    noise_dir = root / "noise_sweep"
    if noise_dir.is_dir():
        for folder in sorted(noise_dir.iterdir()):
            if not folder.is_dir():
                continue
            json_path = folder / "four_region_metrics.json"
            if json_path.is_file():
                row = json.loads(json_path.read_text())
            elif (folder / "results.pt").is_file() or (folder / "summary.json").is_file():
                pct = None
                if folder.name.startswith("noise"):
                    try:
                        pct = int(folder.name.replace("noise", ""))
                    except ValueError:
                        pass
                row = extract_metric_row_from_run(
                    metrics_row_from_run(folder),
                    noise_pct=pct,
                )
            else:
                continue
            if row:
                noise_rows.append(row)

    sparse_dir = root / "sparsity_sweep"
    if sparse_dir.is_dir():
        for folder in sorted(sparse_dir.iterdir()):
            if not folder.is_dir():
                continue
            json_path = folder / "four_region_metrics.json"
            if json_path.is_file():
                row = json.loads(json_path.read_text())
            elif (folder / "results.pt").is_file() or (folder / "summary.json").is_file():
                pct = None
                if folder.name.startswith("sparse"):
                    try:
                        pct = int(folder.name.replace("sparse", ""))
                    except ValueError:
                        pass
                row = extract_metric_row_from_run(
                    metrics_row_from_run(folder),
                    sparse_train_pct=pct,
                )
            else:
                continue
            if row:
                sparsity_rows.append(row)

    noise_rows.sort(key=lambda r: r.get("noise_pct", 0))
    sparsity_rows.sort(key=lambda r: r.get("sparse_train_pct", 0))
    return noise_rows, sparsity_rows
