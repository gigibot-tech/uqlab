"""
Build checkpoint arsenal structure for Step 2.5 UI.

Groups resumable runs (on-disk ``checkpoint.pt``) by model architecture, then
clusters by shared config setup (difference-focused — not by date).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import yaml

from uqlab.evaluation.pipeline.config_diff import (
    arsenal_shared_display,
    arsenal_tracked_flat,
    chip_display_label,
    chip_sort_key,
    chip_tooltip_lines,
    common_tracked_params,
    disambiguate_suffix_ids,
    label_key,
    stable_cluster_fingerprint,
    varying_tracked_key_paths,
)
from uqlab.run_artifacts import RunArtifacts, load_run_directory

MetricTint = Literal["better", "worse", "neutral"]


def _load_run_config(experiments_dir: Path, exp_id: str) -> dict[str, Any]:
    path = experiments_dir / exp_id / "config.yaml"
    if not path.is_file():
        return {}
    try:
        with path.open(encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    except Exception:
        return {}


def _has_checkpoint(experiments_dir: Path, exp_id: str) -> bool:
    return (experiments_dir / exp_id / "results" / "checkpoint.pt").is_file()


def _arch_label_from_config(config: dict[str, Any]) -> str:
    model = config.get("model") or {}
    arch = model.get("architecture", "—")
    if arch == "dinov2_mlp":
        return f"DINOv2-{model.get('dinov2_model', 'small')}"
    if arch == "resnet18_mcdropout":
        return "ResNet18"
    if "resnet50" in str(arch):
        return "ResNet50"
    if arch == "resnet18":
        return "ResNet18"
    return str(arch).replace("_", " ").title()


def _sort_key_created(exp: dict[str, Any]) -> datetime:
    created = exp.get("created_at")
    if isinstance(created, datetime):
        return created
    if isinstance(created, str):
        try:
            return datetime.fromisoformat(created.replace("Z", "+00:00"))
        except ValueError:
            pass
    return datetime.min


def _metric_from_artifacts(artifacts: RunArtifacts | None) -> float | None:
    if artifacts is None or not artifacts.has_data:
        return None
    rows = artifacts.one_vs_rest_auroc
    if not rows:
        return None
    vals: list[float] = []
    for row in rows:
        for key in ("aleatoric_like_auroc", "epistemic_like_auroc", "auroc"):
            val = row.get(key)
            if val is not None:
                try:
                    vals.append(float(val))
                except (TypeError, ValueError):
                    continue
    if not vals:
        return None
    return sum(vals) / len(vals)


def _metric_tint(value: float | None, section_median: float | None) -> MetricTint:
    if value is None or section_median is None:
        return "neutral"
    if value > section_median * 1.001:
        return "better"
    if value < section_median * 0.999:
        return "worse"
    return "neutral"


def _shared_params_line(shared: dict[str, str], *, max_items: int = 10) -> str:
    items = list(shared.items())[:max_items]
    line = " · ".join(f"{k}={v}" for k, v in items)
    if len(shared) > max_items:
        line += f" · +{len(shared) - max_items} more"
    return line


@dataclass(frozen=True)
class CheckpointChip:
    experiment_id: str
    short_id: str
    display_label: str
    status: str
    tooltip_lines: tuple[str, ...]
    metric_value: float | None = None
    metric_tint: MetricTint = "neutral"
    name: str = ""


@dataclass(frozen=True)
class ConfigCluster:
    """Checkpoints that share the same stable training setup; chips differ on sweep axes."""

    cluster_id: str
    shared_params: dict[str, str]
    varying_labels: tuple[str, ...]
    varying_key_paths: tuple[str, ...]
    chips: tuple[CheckpointChip, ...]

    def row_header(self, *, diff_only: bool = True) -> str:
        n = self.n_checkpoints
        count = f"{n} checkpoint" if n == 1 else f"{n} checkpoints"
        if self.varying_labels:
            varies = ", ".join(self.varying_labels)
            if diff_only:
                return f"Varies: {varies} · {count}"
            shared = _shared_params_line(self.shared_params, max_items=6)
            return f"Same: {shared} · differs: {varies}"
        if diff_only:
            return count
        shared = _shared_params_line(self.shared_params, max_items=6)
        return f"Same: {shared}"

    @property
    def summary(self) -> str:
        return self.row_header(diff_only=True)

    @property
    def n_checkpoints(self) -> int:
        return len(self.chips)


@dataclass(frozen=True)
class ModelSection:
    model_label: str
    config_clusters: tuple[ConfigCluster, ...]
    shared_baseline: dict[str, str] = field(default_factory=dict)

    @property
    def n_checkpoints(self) -> int:
        return sum(c.n_checkpoints for c in self.config_clusters)


@dataclass(frozen=True)
class CheckpointArsenal:
    sections: tuple[ModelSection, ...]

    @property
    def n_checkpoints(self) -> int:
        return sum(s.n_checkpoints for s in self.sections)

    @property
    def is_empty(self) -> bool:
        return self.n_checkpoints == 0


def filter_checkpoint_experiments(
    experiments: list[dict[str, Any]],
    *,
    experiments_dir: Path,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for exp in experiments:
        exp_id = str(exp.get("id") or "")
        if not exp_id:
            continue
        if _has_checkpoint(experiments_dir, exp_id):
            out.append(exp)
    return out


def arsenal_cache_key(experiments: list[dict[str, Any]]) -> str:
    if not experiments:
        return "empty"
    times = [_sort_key_created(e) for e in experiments]
    return f"{len(experiments)}:{max(times).isoformat()}"


def filter_arsenal_sections(
    arsenal: CheckpointArsenal,
    *,
    model_label: str = "",
    search_text: str = "",
    sweep_axis: str = "",
    hide_singletons: bool = False,
) -> CheckpointArsenal:
    """Apply structured filters; returns a new arsenal view (does not mutate input)."""
    search = (search_text or "").strip().lower()
    model_pick = (model_label or "").strip()
    axis_pick = (sweep_axis or "").strip()

    filtered_sections: list[ModelSection] = []
    for section in arsenal.sections:
        if model_pick and model_pick != "All" and section.model_label != model_pick:
            continue

        clusters: list[ConfigCluster] = []
        for cluster in section.config_clusters:
            if hide_singletons and cluster.n_checkpoints == 1:
                continue
            if axis_pick and axis_pick != "Any" and axis_pick not in cluster.varying_labels:
                continue
            if search:
                haystacks = [
                    section.model_label.lower(),
                    cluster.row_header().lower(),
                    " ".join(cluster.varying_labels).lower(),
                ]
                haystacks.extend(chip.display_label.lower() for chip in cluster.chips)
                haystacks.extend((chip.name or "").lower() for chip in cluster.chips)
                haystacks.extend(chip.experiment_id.lower() for chip in cluster.chips)
                if not any(search in h for h in haystacks):
                    continue
            clusters.append(cluster)

        if clusters:
            filtered_sections.append(
                ModelSection(
                    model_label=section.model_label,
                    config_clusters=tuple(clusters),
                    shared_baseline=dict(section.shared_baseline),
                )
            )

    return CheckpointArsenal(sections=tuple(filtered_sections))


def collect_sweep_axis_options(arsenal: CheckpointArsenal) -> tuple[str, ...]:
    labels: set[str] = set()
    for section in arsenal.sections:
        for cluster in section.config_clusters:
            labels.update(cluster.varying_labels)
    return tuple(sorted(labels))


def _cluster_experiments(
    model_exps: list[dict[str, Any]],
    configs_by_id: dict[str, dict[str, Any]],
) -> dict[tuple[tuple[str, str], ...], list[dict[str, Any]]]:
    by_fp: dict[tuple[tuple[str, str], ...], list[dict[str, Any]]] = {}
    for exp in model_exps:
        exp_id = str(exp["id"])
        flat = arsenal_tracked_flat(configs_by_id[exp_id])
        fp = stable_cluster_fingerprint(flat)
        by_fp.setdefault(fp, []).append(exp)
    return by_fp


def build_checkpoint_arsenal(
    experiments: list[dict[str, Any]],
    experiments_dir: Path,
) -> CheckpointArsenal:
    """Group checkpoint.pt runs by model → config cluster (shared setup) → chips."""
    ckpt_exps = filter_checkpoint_experiments(experiments, experiments_dir=experiments_dir)
    if not ckpt_exps:
        return CheckpointArsenal(sections=())

    by_model: dict[str, list[dict[str, Any]]] = {}
    configs_by_id: dict[str, dict[str, Any]] = {}
    for exp in ckpt_exps:
        exp_id = str(exp["id"])
        cfg = _load_run_config(experiments_dir, exp_id)
        configs_by_id[exp_id] = cfg
        label = _arch_label_from_config(cfg)
        by_model.setdefault(label, []).append(exp)

    sections: list[ModelSection] = []
    for model_label in sorted(by_model.keys()):
        model_exps = by_model[model_label]
        clusters_by_fp = _cluster_experiments(model_exps, configs_by_id)

        metrics: list[float] = []
        exp_metrics: dict[str, float | None] = {}
        for exp in model_exps:
            exp_id = str(exp["id"])
            run_dir = experiments_dir / exp_id / "results"
            artifacts = load_run_directory(run_dir) if run_dir.is_dir() else None
            mv = _metric_from_artifacts(artifacts)
            exp_metrics[exp_id] = mv
            if mv is not None:
                metrics.append(mv)
        section_median = (sum(metrics) / len(metrics)) if metrics else None

        section_configs = [configs_by_id[str(e["id"])] for e in model_exps]
        shared_baseline = common_tracked_params(section_configs)

        config_clusters: list[ConfigCluster] = []
        for cluster_idx, (_fp, cluster_exps) in enumerate(
            sorted(clusters_by_fp.items(), key=lambda kv: (-len(kv[1]), kv[0]))
        ):
            cluster_configs = [configs_by_id[str(e["id"])] for e in cluster_exps]
            flats = [arsenal_tracked_flat(c) for c in cluster_configs]
            shared = arsenal_shared_display(flats)
            varying_paths = varying_tracked_key_paths(flats)
            varying = tuple(label_key(k) for k in varying_paths)

            sorted_exps = sorted(
                cluster_exps,
                key=lambda e: chip_sort_key(arsenal_tracked_flat(configs_by_id[str(e["id"])])),
            )
            ids = [str(e["id"]) for e in sorted_exps]
            short_map = disambiguate_suffix_ids(ids)

            chips: list[CheckpointChip] = []
            for exp in sorted_exps:
                exp_id = str(exp["id"])
                cfg = configs_by_id[exp_id]
                flat = arsenal_tracked_flat(cfg)
                exp_name = str(exp.get("name") or "")
                mv = exp_metrics.get(exp_id)
                sid = short_map[exp_id]
                chips.append(
                    CheckpointChip(
                        experiment_id=exp_id,
                        short_id=sid,
                        display_label=chip_display_label(
                            flat, varying_paths, short_id=sid
                        ),
                        status=str(exp.get("status") or "—"),
                        tooltip_lines=chip_tooltip_lines(
                            name=exp_name,
                            flat=flat,
                            cluster_flats=flats,
                            experiment_id=exp_id,
                            metric_value=mv,
                        ),
                        metric_value=mv,
                        metric_tint=_metric_tint(mv, section_median),
                        name=exp_name[:40],
                    )
                )

            cluster_id = f"{model_label}:{cluster_idx}"
            config_clusters.append(
                ConfigCluster(
                    cluster_id=cluster_id,
                    shared_params=shared,
                    varying_labels=varying,
                    varying_key_paths=varying_paths,
                    chips=tuple(chips),
                )
            )

        sections.append(
            ModelSection(
                model_label=model_label,
                config_clusters=tuple(config_clusters),
                shared_baseline=shared_baseline,
            )
        )

    return CheckpointArsenal(sections=tuple(sections))
