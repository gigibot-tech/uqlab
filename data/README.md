# Data layer

Three layers — use the highest layer that fits your task.

```
loaders/*  +  dataset_registry     raw per-dataset I/O
       ↓
setup.py                           ExperimentConfig → dataset + SplitSpec
       ↓
packs.py                           SplitSpec → train_dataset + eval packs
```

## When to use what

| Task | Module | Function |
|------|--------|----------|
| Load CIFAR-10N from disk | `dataset_registry` | `load_classification_dataset` |
| Config → splits | `setup.py` | `prepare_experiment_data` |
| Splits → train/eval tensors | `packs.py` | `prepare_run_data_context` |
| Manual notebook (no YAML) | `experiment_loader` | `sample_indices_for_experiment`, `EmbeddingOrganizer` |
| End-to-end images | `image_dataset.py` | `load_image_datasets` (called from `packs.py`) |
| DINOv2 embeddings | `models/feature_extractors.py` | `create_feature_extractor` (called from `packs.py`) |

## Eval pack contract

Every eval pack dict (clean / aleatoric / epistemic / ood) uses the same keys — see `packs.EVAL_PACK_KEYS`:

- `features` — model input tensor (embeddings or images)
- `noisy_labels`, `clean_labels`, `is_noisy`, `original_indices`

## Paper flow

See [`docs/features/PAPER_FLOW.md`](../../docs/features/PAPER_FLOW.md).

- **fit data** → `prepare_experiment_data` then `prepare_run_data_context`
- **fit train** → `models/training.py`

Do **not** call `loaders/*` directly from the runner — go through `setup.py` + `packs.py`.
