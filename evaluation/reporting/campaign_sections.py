"""
Split a smart-grouped campaign into PDF sections (max 2: epistemic + aleatoric sweeps).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from uqlab.evaluation.reporting.sweep_line_plot import (
    SWEEP_KIND_DATASET_SIZE,
    SWEEP_KIND_LABEL_NOISE,
    run_ids_for_experiments,
)

_SECTION_EPISTEMIC = "Epistemic sweep (under-train / dataset size)"
_SECTION_ALEATORIC = "Aleatoric sweep (label noise %)"
_SECTION_FALLBACK = "Campaign sweep"


def experiments_for_perspective(
    experiments: list[dict[str, Any]],
    perspective_id: str,
) -> list[dict[str, Any]]:
    """Filter runs whose names match epistemic or aleatoric sweep patterns."""
    if perspective_id == "epistemic":
        return [
            e
            for e in experiments
            if "fast_epis" in str(e.get("name", "")) or "_under_" in str(e.get("name", ""))
        ]
    if perspective_id == "aleatoric":
        return [
            e
            for e in experiments
            if "fast_alea" in str(e.get("name", "")) or "_noise_" in str(e.get("name", ""))
        ]
    return []


def _plottable_count(
    experiments: list[dict[str, Any]],
    experiments_dir: Path | None,
) -> int:
    return len(run_ids_for_experiments(experiments, experiments_dir=experiments_dir))


@dataclass(frozen=True)
class CampaignSection:
    label: str
    experiments: tuple[dict[str, Any], ...]
    sweep_kind: str | None

    @property
    def n_experiments(self) -> int:
        return len(self.experiments)


def split_campaign_sections(
    experiments: list[dict[str, Any]],
    *,
    experiments_dir: Path | None = None,
    min_runs: int = 2,
) -> tuple[CampaignSection, ...]:
    """
    Partition completed runs into at most two export sections.

    Epistemic (under-train) first, aleatoric (label noise) second.
    Legacy groups with no name match fall back to a single section.
    """
    completed = [e for e in experiments if e.get("status") == "completed"]
    sections: list[CampaignSection] = []

    epistemic = experiments_for_perspective(completed, "epistemic")
    aleatoric = experiments_for_perspective(completed, "aleatoric")

    if _plottable_count(epistemic, experiments_dir) >= min_runs:
        sections.append(
            CampaignSection(
                label=_SECTION_EPISTEMIC,
                experiments=tuple(epistemic),
                sweep_kind=SWEEP_KIND_DATASET_SIZE,
            )
        )
    if _plottable_count(aleatoric, experiments_dir) >= min_runs:
        sections.append(
            CampaignSection(
                label=_SECTION_ALEATORIC,
                experiments=tuple(aleatoric),
                sweep_kind=SWEEP_KIND_LABEL_NOISE,
            )
        )

    if sections:
        return tuple(sections[:2])

    if _plottable_count(completed, experiments_dir) >= min_runs:
        return (
            CampaignSection(
                label=_SECTION_FALLBACK,
                experiments=tuple(completed),
                sweep_kind=None,
            ),
        )

    return ()
