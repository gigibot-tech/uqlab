"""
Data loading and feature extraction for uncertainty classification experiments.

Provides functions for:
- Sampling train/eval splits with controlled class support
- Extracting DINOv2 features from images
- Caching features to disk for faster re-runs
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Dict, List, Sequence, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from uqlab.data.classification_dataset import dataset_clean_labels, dataset_num_classes
from uqlab.data.loaders.cifar10n_loader import CIFAR10NDataset


@dataclass
class SplitSpec:
    """
    Specification of train/eval data splits for uncertainty classification.
    
    Attributes:
        train_indices: Indices for training set
        clean_eval_indices: Indices for clean evaluation samples
        aleatoric_eval_indices: Indices for aleatoric-like samples (noisy labels)
        epistemic_eval_indices: Indices for epistemic-like samples (under-supported classes)
        ood_eval_indices: Indices for OOD eval (withheld-from-train classes; four-region mode)
        under_supported_classes: List of intentionally under-supported class IDs
    """
    train_indices: np.ndarray
    clean_eval_indices: np.ndarray
    aleatoric_eval_indices: np.ndarray
    epistemic_eval_indices: np.ndarray
    under_supported_classes: List[int]
    ood_eval_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))


def sample_indices_for_experiment(
    dataset: Union[datasets.CIFAR10, CIFAR10NDataset],
    *,
    under_supported_classes: Sequence[int],
    under_train_per_class: int,
    regular_train_per_class: int,
    eval_per_group: int,
    seed: int,
    aleatoric_noise_percentage: float = 0.0,
) -> SplitSpec:
    """
    Sample train/eval indices with controlled class support for uncertainty experiments.
    
    Creates three evaluation groups:
    - Clean: Clean samples from well-supported classes
    - Aleatoric-like: Noisy samples (noisy_label != clean_label) from well-supported classes
    - Epistemic-like: Clean samples from intentionally under-supported classes
    
    Args:
        dataset: CIFAR-10 or CIFAR-10N dataset
        under_supported_classes: Classes to intentionally under-support
        under_train_per_class: Number of training samples per under-supported class
        regular_train_per_class: Number of training samples per regular class
        eval_per_group: Number of evaluation samples per group
        seed: Random seed for reproducibility
        aleatoric_noise_percentage: Custom noise percentage (0-100). If > 0, injects
            uniform random label noise instead of using CIFAR-10N noise.
        
    Returns:
        SplitSpec with train and evaluation indices
    """
    # NOTE: Custom noise injection removed - now handled externally before calling this function
    # if aleatoric_noise_percentage > 0:
    #     print(f"\n🎲 Injecting custom uniform noise: {aleatoric_noise_percentage}%")
    #     dataset.inject_custom_noise(noise_percentage=aleatoric_noise_percentage, seed=seed)
    
    rng = np.random.default_rng(seed)

    clean_labels = dataset_clean_labels(dataset)
    raw_noise_mask = getattr(dataset, "noise_mask", None)
    if raw_noise_mask is not None:
        noise_mask = np.asarray(raw_noise_mask, dtype=bool)
    else:
        noise_mask = np.zeros(len(dataset), dtype=bool)
    under_supported_classes = [int(c) for c in under_supported_classes]
    num_classes = dataset_num_classes(dataset)

    train_indices: List[int] = []

    # Epistemic control happens here:
    # - under-supported classes -> `under_train_per_class`
    # - regular classes -> `regular_train_per_class`
    # This is where classes are downsampled for training.
    for cls in range(num_classes):
        cls_all = np.where(clean_labels == cls)[0]
        rng.shuffle(cls_all)
        if cls in under_supported_classes:
            # Under-supported classes use only clean samples.
            cls_clean = cls_all[~noise_mask[cls_all]]
            selected = cls_clean[:under_train_per_class]
        else:
            # Regular classes use the normal training budget.
            if regular_train_per_class is None:
                selected = np.array([], dtype=np.int64)
            else:
                selected = cls_all[:regular_train_per_class]
        train_indices.extend(selected.tolist())

    train_indices = np.array(sorted(set(train_indices)), dtype=np.int64)
    train_mask = np.zeros(len(dataset), dtype=bool)
    train_mask[train_indices] = True

    # Aleatoric is not created here.
    # This function only reads `noise_mask`, which was created earlier in:
    # - `CIFAR10NDataset._load_noisy_labels()` or
    # - `CIFAR10NDataset.inject_custom_noise()`
    #
    # These three pools define the semantic meaning of the benchmark:
    #
    # - clean_eval_pool:
    #     clean samples from regular (well-supported) classes
    # - aleatoric_eval_pool:
    #     samples already marked noisy in `noise_mask`, from regular classes
    # - epistemic_eval_pool:
    #     clean samples from under-supported classes
    #
    # So:
    # - aleatoric is sourced from label noise already present on the dataset
    # - epistemic is sourced from reduced training support for selected classes
    under_mask = np.isin(clean_labels, np.asarray(under_supported_classes))
    non_under_mask = ~under_mask
    clean_mask = ~noise_mask

    clean_eval_pool = np.where(non_under_mask & clean_mask & ~train_mask)[0]
    aleatoric_eval_pool = np.where(non_under_mask & noise_mask & ~train_mask)[0]
    epistemic_eval_pool = np.where(under_mask & clean_mask & ~train_mask)[0]

    # Sample evaluation sets - use min(requested, available) to handle edge cases
    rng.shuffle(clean_eval_pool)
    rng.shuffle(aleatoric_eval_pool)
    rng.shuffle(epistemic_eval_pool)

    # Take up to eval_per_group samples, but use whatever is available
    clean_eval_indices = clean_eval_pool[:min(eval_per_group, len(clean_eval_pool))]
    aleatoric_eval_indices = aleatoric_eval_pool[:min(eval_per_group, len(aleatoric_eval_pool))]
    epistemic_eval_indices = epistemic_eval_pool[:min(eval_per_group, len(epistemic_eval_pool))]
    
    # Log warnings if we got fewer samples than requested
    import logging
    logger = logging.getLogger(__name__)
    
    if len(clean_eval_indices) < eval_per_group:
        logger.warning(
            f"⚠️  Clean eval pool: requested {eval_per_group}, got {len(clean_eval_indices)} "
            f"(pool size: {len(clean_eval_pool)})"
        )
    from uqlab.data.benchmark_axes import (
        expects_aleatoric_eval,
        expects_epistemic_eval,
    )

    if len(aleatoric_eval_indices) < eval_per_group:
        if expects_aleatoric_eval(aleatoric_noise_percentage):
            logger.warning(
                f"⚠️  Aleatoric eval pool: requested {eval_per_group}, "
                f"got {len(aleatoric_eval_indices)} (pool size: {len(aleatoric_eval_pool)}). "
                f"Label noise is {aleatoric_noise_percentage}% but no noisy eval samples were found."
            )
        else:
            logger.info(
                "ℹ️  Aleatoric eval skipped (0% label noise — epistemic/clean benchmark)."
            )
    if len(epistemic_eval_indices) < eval_per_group:
        if expects_epistemic_eval(
            under_supported_classes,
            under_train_per_class=under_train_per_class,
            regular_train_per_class=regular_train_per_class,
        ):
            logger.warning(
                f"⚠️  Epistemic eval pool: requested {eval_per_group}, "
                f"got {len(epistemic_eval_indices)} (pool size: {len(epistemic_eval_pool)})"
            )
        else:
            logger.info(
                "ℹ️  Epistemic eval skipped (balanced training — aleatoric/noise benchmark)."
            )

    return SplitSpec(
        train_indices=train_indices,
        clean_eval_indices=clean_eval_indices,
        aleatoric_eval_indices=aleatoric_eval_indices,
        epistemic_eval_indices=epistemic_eval_indices,
        under_supported_classes=under_supported_classes,
    )


@torch.no_grad()
def extract_features_for_indices(
    dataset: CIFAR10NDataset,
    indices: Sequence[int],
    *,
    dinov2_model: str,
    batch_size: int,
    device: torch.device,
    use_untrained_resnet: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Extract features for specified dataset indices.
    
    Args:
        dataset: CIFAR-10N dataset
        indices: Indices to extract features for
        dinov2_model: DINOv2 model size ('small', 'base', 'large', 'giant') - ignored if use_untrained_resnet=True
        batch_size: Batch size for feature extraction
        device: Device to run on
        use_untrained_resnet: If True, use untrained ResNet-50 instead of DINOv2
        
    Returns:
        Dictionary containing:
            - features: Extracted features [N, feature_dim]
            - noisy_labels: Training labels [N]
            - clean_labels: Ground truth labels [N]
            - is_noisy: Boolean mask [N]
            - original_indices: Original dataset indices [N]
    """
    subset = Subset(dataset, list(indices))
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)

    if use_untrained_resnet:
        # Use untrained ResNet-50 as feature extractor
        import torchvision.models as models
        import torch.nn as nn
        
        resnet = models.resnet50(weights=None)  # Untrained weights
        # Remove final classification layer to get features
        model = nn.Sequential(*list(resnet.children())[:-1])
        model = model.to(device)
        model.eval()
        
        def extract_batch_features(images):
            features = model(images)
            return features.squeeze(-1).squeeze(-1)  # [B, 2048]
    else:
        # Use DINOv2 as before
        try:
            from uqlab.models.backbones.dinov2_backbone import create_dinov2_model
        except Exception as exc:
            raise RuntimeError(
                "DINOv2 feature extraction requires the DINOv2 dependencies. "
                "Please run this in the project venv where `transformers` is available."
            ) from exc

        model = create_dinov2_model(
            model_name=dinov2_model,
            num_classes=10,
            dropout_rate=0.0,
            mc_dropout=False,
            freeze_backbone=True,
        ).to(device)
        model.eval()
        
        def extract_batch_features(images):
            return model.extract_features(images)  # Returns [B, feature_dim] with CLS token

    all_features: List[torch.Tensor] = []
    all_noisy_labels: List[torch.Tensor] = []
    all_clean_labels: List[torch.Tensor] = []
    all_is_noisy: List[torch.Tensor] = []

    for batch in loader:
        # Handle both CIFAR10 (2 values) and CIFAR10NDataset (4 values)
        if len(batch) == 2:
            # Raw CIFAR10: (images, labels)
            images, labels = batch
            noisy_labels = labels  # No noise, so noisy = clean
            clean_labels = labels
            is_noisy = torch.zeros(len(labels), dtype=torch.bool)
        else:
            # CIFAR10NDataset: (images, noisy_labels, clean_labels, is_noisy)
            images, noisy_labels, clean_labels, is_noisy = batch
        
        features = extract_batch_features(images.to(device))
        all_features.append(features.cpu())
        all_noisy_labels.append(noisy_labels.cpu())
        all_clean_labels.append(clean_labels.cpu())
        all_is_noisy.append(is_noisy.cpu())

    return {
        "features": torch.cat(all_features, dim=0),
        "noisy_labels": torch.cat(all_noisy_labels, dim=0),
        "clean_labels": torch.cat(all_clean_labels, dim=0),
        "is_noisy": torch.cat(all_is_noisy, dim=0).bool(),
        "original_indices": torch.as_tensor(indices, dtype=torch.long),
    }


def maybe_load_or_compute_feature_cache(
    dataset: CIFAR10NDataset,
    indices: Sequence[int],
    *,
    cache_file: Path,
    dinov2_model: str,
    batch_size: int,
    device: torch.device,
    use_untrained_resnet: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Load features from cache or compute and cache them.
    
    Args:
        dataset: CIFAR-10N dataset
        indices: Indices to extract features for
        cache_file: Path to cache file
        dinov2_model: DINOv2 model size (ignored if use_untrained_resnet=True)
        batch_size: Batch size for feature extraction
        device: Device to run on
        use_untrained_resnet: If True, use untrained ResNet-50 instead of DINOv2
        
    Returns:
        Dictionary with features and labels
    """
    if cache_file.exists():
        print(f"Loading cached features from {cache_file}")
        return torch.load(cache_file, map_location="cpu", weights_only=False)

    print(f"Computing features (will cache to {cache_file})")
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    payload = extract_features_for_indices(
        dataset,
        indices,
        dinov2_model=dinov2_model,
        batch_size=batch_size,
        device=device,
        use_untrained_resnet=use_untrained_resnet,
    )
    torch.save(payload, cache_file)
    return payload


def build_feature_cache_path(
    cache_dir: Path,
    indices: Sequence[int],
    *,
    noise_type: str,
    dinov2_model: str,
    use_untrained_resnet: bool = False,
) -> Path:
    """
    Build a stable feature-cache path from the selected data indices.

    The hash keeps the filename short while making the cache reusable across
    result directories whenever the same split and backbone are used again.
    """
    index_bytes = np.asarray(indices, dtype=np.int64).tobytes()
    index_hash = hashlib.sha1(index_bytes).hexdigest()[:12]
    
    if use_untrained_resnet:
        model_name = "resnet50_untrained"
    else:
        model_name = dinov2_model
    
    return cache_dir / f"features_{noise_type}_{model_name}_n{len(indices)}_{index_hash}.pt"


def train_feature_model(
    train_dataset,
    *,
    device: torch.device,
    num_classes: int,
    hidden_dim: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
):
    """
    Train an embedding-space classifier with dropout.
    
    Args:
        train_dataset: EmbeddingDataset with pre-extracted DINOv2 embeddings
        device: Device to train on
        num_classes: Number of output classes
        hidden_dim: Hidden layer dimension
        dropout: Dropout probability
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        weight_decay: Weight decay for regularization
        
    Returns:
        Trained EmbeddingDropoutMLP model
    """
    from .models import EmbeddingDropoutMLP
    import torch.nn as nn
    from torch.utils.data import DataLoader
    
    model = EmbeddingDropoutMLP(
        input_dim=int(train_dataset.features.shape[1]),
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)

    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        if (epoch + 1) % max(1, epochs // 3) == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch + 1}/{epochs}, loss={total_loss / max(1, len(loader)):.4f}")

    return model


class EmbeddingOrganizer:
    """
    Organizes DINOv2 embedding extraction and split management for uncertainty classification.
    
    This class encapsulates the complex logic of:
    1. Loading/computing embeddings (768-dim vectors) for all samples (train + eval)
    2. Building an index mapping from dataset indices to embedding positions
    3. Extracting embeddings for specific splits (train, clean_eval, aleatoric_eval, epistemic_eval)
    
    Why this class exists:
    ----------------------
    The original code had confusing nested logic with:
    - union_indices concatenation
    - index_to_pos dictionary mapping
    - gather() closure function
    - Multiple pack extractions
    
    This class makes the data flow explicit and testable.
    
    Example Usage:
    --------------
    ```python
    organizer = EmbeddingOrganizer(
        dataset=cifar10n_dataset,
        split_spec=split_spec,
        feature_cache_dir=Path("./cache"),
        noise_type="worse_label",
        dinov2_model="dinov2_vits14",
        batch_size=256,
        device=torch.device("cuda"),
    )
    
    # Load embeddings once
    organizer.load_or_compute_features()
    
    # Extract organized packs
    train_pack = organizer.get_train_pack()
    clean_eval_pack = organizer.get_clean_eval_pack()
    ```
    """
    
    def __init__(
        self,
        dataset: CIFAR10NDataset,
        split_spec: SplitSpec,
        feature_cache_dir: Path,
        noise_type: str,
        dinov2_model: str,
        batch_size: int,
        device: torch.device,
    ):
        """
        Initialize the embedding organizer.
        
        Args:
            dataset: CIFAR-10N dataset
            split_spec: Train/eval split specification
            feature_cache_dir: Directory for embedding caching
            noise_type: CIFAR-10N noise type
            dinov2_model: DINOv2 model size
            batch_size: Batch size for embedding extraction
            device: Device to run on
        """
        self.dataset = dataset
        self.split_spec = split_spec
        self.feature_cache_dir = feature_cache_dir
        self.noise_type = noise_type
        self.dinov2_model = dinov2_model
        self.batch_size = batch_size
        self.device = device
        
        # Will be populated by load_or_compute_features()
        self._payload: Dict[str, torch.Tensor] | None = None
        self._index_to_pos: Dict[int, int] | None = None
    
    def load_or_compute_features(self) -> None:
        """
        Load embeddings from cache or compute them.
        
        This method:
        1. Concatenates all indices (train + all eval groups)
        2. Builds cache path based on indices hash
        3. Loads from cache or computes embeddings
        4. Creates index-to-position mapping for fast lookup
        """
        # Step 1: Combine all indices we need features for
        union_indices = np.concatenate([
            self.split_spec.train_indices,
            self.split_spec.clean_eval_indices,
            self.split_spec.aleatoric_eval_indices,
            self.split_spec.epistemic_eval_indices,
            self.split_spec.ood_eval_indices,
        ])
        
        # Step 2: Build cache path
        cache_file = build_feature_cache_path(
            self.feature_cache_dir,
            union_indices.tolist(),
            noise_type=self.noise_type,
            dinov2_model=self.dinov2_model,
            use_untrained_resnet=False,  # EmbeddingOrganizer doesn't support ResNet yet
        )
        
        # Step 3: Load or compute features
        self._payload = maybe_load_or_compute_feature_cache(
            self.dataset,
            union_indices.tolist(),
            cache_file=cache_file,
            dinov2_model=self.dinov2_model,
            batch_size=self.batch_size,
            device=self.device,
        )
        
        # Step 4: Build index mapping for fast lookup
        # Maps: dataset_index -> position_in_payload
        original_indices = self._payload["original_indices"].numpy()
        self._index_to_pos = {
            int(idx): pos
            for pos, idx in enumerate(original_indices.tolist())
        }
    
    def _gather(self, indices: np.ndarray) -> Dict[str, torch.Tensor]:
        """
        Extract embeddings for specific indices from the loaded payload.
        
        Args:
            indices: Dataset indices to extract
            
        Returns:
            Dictionary with embeddings (768-dim), labels, and metadata for the indices
            
        Raises:
            RuntimeError: If embeddings haven't been loaded yet
        """
        if self._payload is None or self._index_to_pos is None:
            raise RuntimeError(
                "Embeddings not loaded. Call load_or_compute_features() first."
            )

        if len(indices) == 0:
            feat_dim = int(self._payload["features"].shape[1])
            return {
                "features": torch.empty((0, feat_dim), dtype=self._payload["features"].dtype),
                "noisy_labels": torch.empty(0, dtype=torch.long),
                "clean_labels": torch.empty(0, dtype=torch.long),
                "is_noisy": torch.empty(0, dtype=torch.bool),
                "original_indices": torch.empty(0, dtype=torch.long),
            }

        # Map dataset indices to payload positions
        positions = torch.as_tensor(
            [self._index_to_pos[int(idx)] for idx in indices],
            dtype=torch.long
        )
        
        # Extract data at those positions
        return {
            "features": self._payload["features"][positions],
            "noisy_labels": self._payload["noisy_labels"][positions],
            "clean_labels": self._payload["clean_labels"][positions],
            "is_noisy": self._payload["is_noisy"][positions],
            "original_indices": self._payload["original_indices"][positions],
        }
    
    def get_train_pack(self) -> Dict[str, torch.Tensor]:
        """Extract training data pack."""
        return self._gather(self.split_spec.train_indices)
    
    def get_clean_eval_pack(self) -> Dict[str, torch.Tensor]:
        """Extract clean evaluation data pack."""
        return self._gather(self.split_spec.clean_eval_indices)
    
    def get_aleatoric_eval_pack(self) -> Dict[str, torch.Tensor]:
        """Extract aleatoric evaluation data pack."""
        return self._gather(self.split_spec.aleatoric_eval_indices)
    
    def get_epistemic_eval_pack(self) -> Dict[str, torch.Tensor]:
        """Extract epistemic evaluation data pack."""
        return self._gather(self.split_spec.epistemic_eval_indices)

    def get_ood_eval_pack(self) -> Dict[str, torch.Tensor]:
        """Extract OOD evaluation data pack (four-region mode)."""
        return self._gather(self.split_spec.ood_eval_indices)


# Made with Bob
