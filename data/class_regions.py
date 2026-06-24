"""Four-region partition (noisy / sparse / clean / OOD)."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from uqlab.data.classification_dataset import dataset_clean_labels, dataset_num_classes
from uqlab.data.experiment_loader import SplitSpec

REGION_NOISY = "noisy"
REGION_SPARSE = "sparse"
REGION_CLEAN = "clean"
REGION_OOD = "ood"

ALL_REGIONS = (REGION_NOISY, REGION_SPARSE, REGION_CLEAN, REGION_OOD)

# CIFAR-10 classes 0–9 in four blocks of two.
DEFAULT_FOUR_REGION_PRESET: dict[str, dict[str, Any]] = {
    REGION_NOISY: {"classes": [0, 1, 2, 3], "label_flip_pct": 30.0, "train_fraction": 1.0},
    REGION_SPARSE: {"classes": [4, 5], "train_fraction": 0.10},
    REGION_CLEAN: {"classes": [6, 7], "train_fraction": 1.0},
    REGION_OOD: {"classes": [8, 9], "train_fraction": 0.0},
}


def normalize_class_regions(raw: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Deep-copy and normalize region specs from workflow or YAML."""
    if not raw:
        return {k: dict(v) for k, v in DEFAULT_FOUR_REGION_PRESET.items()}
    out: dict[str, dict[str, Any]] = {}
    for name in ALL_REGIONS:
        spec = dict(raw.get(name) or DEFAULT_FOUR_REGION_PRESET[name])
        spec["classes"] = [int(c) for c in spec.get("classes") or []]
        if "train_fraction" in spec:
            spec["train_fraction"] = float(spec["train_fraction"])
        if "label_flip_pct" in spec:
            spec["label_flip_pct"] = float(spec["label_flip_pct"])
        out[name] = spec
    return out


def validate_class_regions(
    class_regions: Mapping[str, Mapping[str, Any]],
    *,
    num_classes: int = 10,
) -> None:
    """Every class 0..num_classes-1 appears in exactly one region."""
    seen: dict[int, str] = {}
    for region_name, spec in class_regions.items():
        if region_name not in ALL_REGIONS:
            raise ValueError(f"Unknown region {region_name!r}; expected one of {ALL_REGIONS}")
        for cls in spec.get("classes") or []:
            c = int(cls)
            if c < 0 or c >= num_classes:
                raise ValueError(f"Invalid class id {c} for region {region_name!r}")
            if c in seen:
                raise ValueError(f"Class {c} assigned to both {seen[c]!r} and {region_name!r}")
            seen[c] = region_name
    missing = [c for c in range(num_classes) if c not in seen]
    if missing:
        raise ValueError(f"Classes not assigned to any region: {missing}")


def inject_class_label_noise(
    dataset: object,
    classes: Sequence[int],
    flip_pct: float,
    *,
    seed: int = 42,
) -> None:
    """Flip ``flip_pct`` percent of labels within *classes* only (aleatoric region)."""
    if flip_pct <= 0 or not classes:
        return

    clean = np.asarray(dataset_clean_labels(dataset))
    num_classes = dataset_num_classes(dataset)
    class_set = {int(c) for c in classes}

    noisy = (
        np.asarray(dataset.noisy_labels).copy()
        if getattr(dataset, "noisy_labels", None) is not None
        else clean.copy()
    )
    mask = (
        np.asarray(dataset.noise_mask, dtype=bool).copy()
        if getattr(dataset, "noise_mask", None) is not None
        else np.zeros(len(clean), dtype=bool)
    )

    rng = np.random.default_rng(seed)
    for cls in class_set:
        cls_idx = np.where(clean == cls)[0]
        if len(cls_idx) == 0:
            continue
        n_flip = int(round(len(cls_idx) * float(flip_pct) / 100.0))
        if n_flip <= 0:
            continue
        flip_idx = rng.choice(cls_idx, size=min(n_flip, len(cls_idx)), replace=False)
        for idx in flip_idx:
            original = int(clean[idx])
            wrong = [c for c in range(num_classes) if c != original]
            noisy[idx] = int(rng.choice(wrong))
            mask[idx] = True

    dataset.noisy_labels = noisy
    dataset.noise_mask = mask
    dataset.noise_rate = float(mask.mean())


def apply_region_noise(
    dataset: object,
    class_regions: Mapping[str, Mapping[str, Any]],
    *,
    seed: int,
) -> None:
    """Apply per-region label flip (noisy block only)."""
    noisy_spec = class_regions.get(REGION_NOISY) or {}
    flip = float(noisy_spec.get("label_flip_pct") or 0.0)
    classes = noisy_spec.get("classes") or []
    inject_class_label_noise(dataset, classes, flip, seed=seed)


def _class_to_region(class_regions: Mapping[str, Mapping[str, Any]]) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for region_name, spec in class_regions.items():
        for cls in spec.get("classes") or []:
            mapping[int(cls)] = region_name
    return mapping


def sample_indices_for_four_region(
    dataset: object,
    class_regions: Mapping[str, Mapping[str, Any]],
    *,
    regular_train_per_class: int,
    eval_per_group: int,
    seed: int,
) -> SplitSpec:
    """
    Build train/eval splits from four explicit class regions.

    Maps to eval packs: noisy→aleatoric_like, sparse→epistemic_like, clean→clean, ood→ood.
    """
    validate_class_regions(class_regions, num_classes=dataset_num_classes(dataset))
    regions = normalize_class_regions(class_regions)
    c2r = _class_to_region(regions)

    rng = np.random.default_rng(seed)
    clean_labels = dataset_clean_labels(dataset)
    raw_noise_mask = getattr(dataset, "noise_mask", None)
    noise_mask = (
        np.asarray(raw_noise_mask, dtype=bool)
        if raw_noise_mask is not None
        else np.zeros(len(clean_labels), dtype=bool)
    )
    num_classes = dataset_num_classes(dataset)

    train_parts: list[int] = []
    clean_pool: list[int] = []
    aleatoric_pool: list[int] = []
    epistemic_pool: list[int] = []
    ood_pool: list[int] = []

    for cls in range(num_classes):
        region = c2r[cls]
        spec = regions[region]
        cls_all = np.where(clean_labels == cls)[0]
        rng.shuffle(cls_all)
        train_fraction = float(spec.get("train_fraction", 1.0))

        if region == REGION_OOD or train_fraction <= 0.0:
            ood_pool.extend(cls_all.tolist())
            continue

        if train_fraction >= 1.0:
            n_train = min(len(cls_all), int(regular_train_per_class))
        else:
            n_train = max(0, int(round(len(cls_all) * train_fraction)))

        train_sel = cls_all[:n_train]
        held_out = cls_all[n_train:]
        train_parts.extend(train_sel.tolist())

        clean_mask = ~noise_mask
        if region == REGION_NOISY:
            aleatoric_pool.extend([int(i) for i in held_out if noise_mask[i]])
            clean_pool.extend([int(i) for i in held_out if clean_mask[i]])
        elif region == REGION_SPARSE:
            epistemic_pool.extend([int(i) for i in held_out if clean_mask[i]])
        elif region == REGION_CLEAN:
            clean_pool.extend([int(i) for i in held_out if clean_mask[i]])

    train_indices = np.array(sorted(set(train_parts)), dtype=np.int64)

    def _sample(pool: list[int]) -> np.ndarray:
        arr = np.array(pool, dtype=np.int64)
        if len(arr) == 0:
            return arr
        rng.shuffle(arr)
        return arr[: min(eval_per_group, len(arr))]

    sparse_classes = [int(c) for c in regions[REGION_SPARSE].get("classes") or []]

    return SplitSpec(
        train_indices=train_indices,
        clean_eval_indices=_sample(clean_pool),
        aleatoric_eval_indices=_sample(aleatoric_pool),
        epistemic_eval_indices=_sample(epistemic_pool),
        ood_eval_indices=_sample(ood_pool),
        under_supported_classes=sparse_classes,
    )
