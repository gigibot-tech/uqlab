"""
DualXDA (DualDA) triage utilities.

This module connects the thesis-level definition

  T_i(x) = lambda_i * <f(x_i), f(x)>

to an executable approximation using the `dualxda` package:
  - DualDA trains a sparse SVM surrogate on the penultimate features.
  - DualDA.attribute returns signed training-point attributions for a given target.

From those traces we compute the "tau" diagnostics:
  mass(x)   = sum_i |T_i(x)|
  signal(x) = |sum_i T_i(x)|
  dom(x)    = max_i |T_i(x)| / (mass(x) + eps)

And triage each x into (Void / Pile / Intruder) via thresholds.

Calibration: with DualDA over all training points, max_i|T_i|/sum_i|T_i| is usually far below 0.8;
use coherence = |sum T|/(sum|T|+eps) for the Pile vs non-Pile split unless you tune raw |sum T| thresholds.

Helpers: :func:`print_dualxda_batch_debug`, :func:`format_dualxda_debug_lines`, and
:meth:`DualXDATracer.debug_print_batch` for quick inspection of τ scalars and top ± attributions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


@dataclass(frozen=True)
class AxiomThresholds:
    tau_mass: float
    tau_signal: float
    tau_dom: float
    eps: float = 1e-12
    # "absolute_signal": Pile iff signal <= tau_signal (same units as DualDA attributions).
    # "coherence": Pile iff (signal / (mass+eps)) <= tau_signal — scale-free; use when
    # raw |sum T| is never tiny (typical for full N-train DualDA traces).
    pile_metric: str = "coherence"


def infer_classifier_layer_name(model: torch.nn.Module) -> str:
    """
    Best-effort guess of the classifier layer name for DualDA hooks.

    DualDA expects the *final* classification layer (typically the last Linear) so it can
    capture its input features and train an SVM surrogate on the penultimate representations.

    We prioritize common patterns used in this repo:
      - ResNet-style heads: `model.fc`
      - Sequential heads: `model.classifier.<last_linear_idx>`
    and fall back to the last `nn.Linear` discovered in `named_modules()`.
    """
    if hasattr(model, "fc") and isinstance(getattr(model, "fc"), nn.Linear):
        return "fc"

    if hasattr(model, "classifier"):
        clf = getattr(model, "classifier")
        if isinstance(clf, nn.Linear):
            return "classifier"
        if isinstance(clf, nn.Sequential):
            for idx in range(len(clf) - 1, -1, -1):
                if isinstance(clf[idx], nn.Linear):
                    return f"classifier.{idx}"

    last_linear_name = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            last_linear_name = name
    if last_linear_name:
        return last_linear_name

    raise ValueError("Could not infer classifier layer name for DualDA (no nn.Linear modules found).")


def _infer_labels_from_dataset(ds: Any) -> Optional[torch.Tensor]:
    """
    Best-effort extraction of training labels without iterating images.

    Supports:
      - torchvision datasets with .targets
      - CIFAR10NDataset with .noisy_labels / .clean_labels (we prefer noisy_labels if present)
      - CIFAR10NLabelView delegating to base_dataset
    """
    # CIFAR-10N base dataset
    if hasattr(ds, "noisy_labels"):
        try:
            return torch.as_tensor(ds.noisy_labels, dtype=torch.long)
        except Exception:
            pass
    if hasattr(ds, "targets"):
        try:
            return torch.as_tensor(ds.targets, dtype=torch.long)
        except Exception:
            pass

    # CIFAR10NLabelView delegates via __getattr__ to base_dataset
    if hasattr(ds, "base_dataset"):
        return _infer_labels_from_dataset(ds.base_dataset)

    return None


@torch.no_grad()
def compute_axiom_signals(attr: torch.Tensor, th: AxiomThresholds) -> Dict[str, torch.Tensor]:
    """
    Stateless: (mass, signal, dominance, argmax index) per row of attr.

    attr: [B, N_train] signed DualDA attributions.
    """
    abs_attr = attr.abs()
    mass = abs_attr.sum(dim=1)
    signal = attr.sum(dim=1).abs()
    dom_val, dom_idx = abs_attr.max(dim=1)
    dominance = dom_val / (mass + th.eps)

    return {
        "mass": mass,
        "signal": signal,
        "dominance": dominance,
        "dom_idx": dom_idx,
        "dom_val": dom_val,
    }


@torch.no_grad()
def compute_axiom_uncertainty_scores(attr: torch.Tensor, th: AxiomThresholds) -> Dict[str, torch.Tensor]:
    """
    Compute 3 uncertainty scores from DualXDA attributions for AURC analysis.
    
    These scores quantify different aspects of spurious features and data quality:
    
    1. **Absolute Signal Score** (|sum T_i|):
       - Measures coherence of training influences
       - High value = consistent training support (low uncertainty)
       - Low value = conflicting training influences (high uncertainty)
       - Use inverted for AURC: uncertainty = 1 / (|sum T_i| + eps)
    
    2. **Triage-Based Score**:
       - Combines axiom assignments into a single uncertainty metric
       - Void (no support) = highest uncertainty
       - Pile (conflicting) = medium uncertainty
       - Intruder (spurious) = medium-high uncertainty
       - Other (clean) = lowest uncertainty
    
    3. **Spurious Feature Score** (dominance × signal_weakness):
       - Identifies samples relying on spurious correlations
       - High dominance + low coherence = spurious feature
       - Combines structural (dominance) and semantic (coherence) signals
    
    Args:
        attr: [B, N_train] signed DualDA attributions
        th: AxiomThresholds for triage logic
        
    Returns:
        Dictionary with 3 uncertainty scores, each in range suitable for AURC ranking
    """
    sig = compute_axiom_signals(attr, th)
    tri = compute_triage(attr, th)
    
    mass = sig["mass"]
    signal = sig["signal"]
    dominance = sig["dominance"]
    coherence = tri["coherence"]
    
    # Score 1: Absolute Signal Uncertainty
    # Invert signal so high uncertainty = low signal (conflicting influences)
    # Add small constant to avoid division by zero
    signal_uncertainty = 1.0 / (signal + th.eps)
    # Normalize to reasonable range [0, 1] using sigmoid-like transform
    signal_uncertainty = torch.tanh(signal_uncertainty / 10.0)
    
    # Score 2: Triage-Based Uncertainty
    # Assign uncertainty levels based on axiom classification
    triage_uncertainty = torch.zeros_like(mass)
    triage_uncertainty[tri["is_void"]] = 1.0      # Highest: no training support
    triage_uncertainty[tri["is_pile"]] = 0.6      # Medium: conflicting influences
    triage_uncertainty[tri["is_intruder"]] = 0.7  # Medium-high: spurious features
    # Remaining samples (clean) get 0.0 (lowest uncertainty)
    
    # Score 3: Spurious Feature Score
    # High dominance (single strong influence) + low coherence (weak overall signal)
    # indicates reliance on spurious correlation
    signal_weakness = 1.0 - coherence  # Low coherence = high weakness
    spurious_score = dominance * signal_weakness
    
    return {
        "signal_uncertainty": signal_uncertainty,
        "triage_uncertainty": triage_uncertainty,
        "spurious_score": spurious_score,
        # Also return components for analysis
        "mass": mass,
        "signal": signal,
        "coherence": coherence,
        "dominance": dominance,
    }


@torch.no_grad()
def compute_triage(
    attr: torch.Tensor,
    th: AxiomThresholds,
    *,
    targets: Optional[torch.Tensor] = None,
    train_labels: Optional[torch.Tensor] = None,
    intruder_rule: str = "structural",
) -> Dict[str, torch.Tensor]:
    """
    Stateless twin of ``DualXDATracer.triage`` for notebooks and scripts that already
    have an attribution tensor (e.g. from ``explainer.explain``) without rebuilding DualDA.
    """
    sig = compute_axiom_signals(attr, th)

    mass = sig["mass"]
    signal = sig["signal"]
    dominance = sig["dominance"]
    dom_idx = sig["dom_idx"]
    coherence = signal / (mass + th.eps)

    metric = (th.pile_metric or "coherence").lower()
    if metric == "absolute_signal":
        pile_score = signal
    elif metric in ("coherence", "coh"):
        pile_score = coherence
    else:
        raise ValueError(f"Unknown pile_metric={th.pile_metric!r}; use 'coherence' or 'absolute_signal'.")

    is_void = mass < th.tau_mass
    is_pile = (~is_void) & (pile_score <= th.tau_signal)
    intruder_candidate = (~is_void) & (~is_pile) & (dominance >= th.tau_dom)
    is_intruder = intruder_candidate

    out: Dict[str, torch.Tensor] = {
        **sig,
        "coherence": coherence,
        "pile_score": pile_score,
        "is_void": is_void,
        "is_pile": is_pile,
        "is_intruder": is_intruder,
        "intruder_candidate": intruder_candidate,
    }

    if targets is not None and train_labels is not None:
        dom_y = train_labels[dom_idx.cpu()].to(targets.device)
        dom_label_conflict = dom_y != targets
        dom_attr = attr.gather(1, dom_idx.view(-1, 1)).squeeze(1)
        dom_supports_target = dom_attr > 0

        out["dom_label_conflict"] = dom_label_conflict
        out["dom_supports_target"] = dom_supports_target

        if str(intruder_rule).lower() == "inverse_mirror":
            is_intruder = intruder_candidate & dom_label_conflict & dom_supports_target
            out["is_intruder"] = is_intruder
    return out


def axiom_name_from_triage(tri: Dict[str, torch.Tensor], row: int) -> str:
    """MECE-ish label for one row: void | pile | intruder | other."""
    if bool(tri["is_void"][row].item()):
        return "void"
    if bool(tri["is_pile"][row].item()):
        return "pile"
    if bool(tri["is_intruder"][row].item()):
        return "intruder"
    return "other"


def _topk_positive_negative(
    scores_1d: torch.Tensor, k: int
) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    """Return up to k (train_index, value) pairs for largest positive and most negative entries."""
    s = scores_1d.detach().flatten()
    pos_mask = s > 0
    neg_mask = s < 0
    pos_idx = torch.nonzero(pos_mask, as_tuple=False).view(-1)
    neg_idx = torch.nonzero(neg_mask, as_tuple=False).view(-1)

    pos_out: List[Tuple[int, float]] = []
    if pos_idx.numel() > 0:
        pv = s[pos_idx]
        tk = min(k, int(pv.numel()))
        vals, sub = torch.topk(pv, tk)
        for j in range(tk):
            gi = int(pos_idx[int(sub[j])].item())
            pos_out.append((gi, float(vals[j].item())))

    neg_out: List[Tuple[int, float]] = []
    if neg_idx.numel() > 0:
        nv = s[neg_idx]
        tk = min(k, int(nv.numel()))
        vals, sub = torch.topk(-nv, tk)
        for j in range(tk):
            gi = int(neg_idx[int(sub[j])].item())
            neg_out.append((gi, float(s[gi].item())))

    return pos_out, neg_out


def topk_positive_negative(
    scores_1d: torch.Tensor, k: int
) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    """
    Paper-aligned ranking: up to ``k`` largest positive :math:`T_i` and ``k`` most negative :math:`T_i`.

    Contrast with ``torch.topk(scores, k)``, which returns the ``k`` largest *algebraic* values
    (often dominated by positives; strong negatives may be absent from that single list).
    """
    return _topk_positive_negative(scores_1d, k)


def format_dualxda_debug_lines(
    attr: torch.Tensor,
    tri: Dict[str, torch.Tensor],
    *,
    batch_indices: Optional[Sequence[int]] = None,
    k: int = 5,
) -> List[str]:
    """
    Human-readable lines: mass, coherence, dominance, axiom, top-k pos/neg train indices.

    attr: [B, N_train]; tri: output of ``compute_triage`` or ``DualXDATracer.triage``.
    """
    if attr.dim() != 2:
        raise ValueError(f"attr must be [B, N_train], got shape {tuple(attr.shape)}")
    B = int(attr.shape[0])
    idxs: Sequence[int] = batch_indices if batch_indices is not None else range(B)
    lines: List[str] = []
    for i in idxs:
        if i < 0 or i >= B:
            continue
        name = axiom_name_from_triage(tri, i)
        mass = float(tri["mass"][i].item())
        coh = float(tri["coherence"][i].item())
        dom = float(tri["dominance"][i].item())
        dix = int(tri["dom_idx"][i].item())
        lines.append(
            f"[sample {i}] axiom={name}  mass={mass:.6g}  coherence={coh:.6g}  "
            f"dominance={dom:.6g}  dom_idx={dix}"
        )
        pos, neg = _topk_positive_negative(attr[i], k)
        lines.append(f"    top {k} positive T_i (idx, val): {pos}")
        lines.append(f"    top {k} negative T_i (idx, val): {neg}")
    return lines


def print_dualxda_batch_debug(
    attr: torch.Tensor,
    tri: Dict[str, torch.Tensor],
    *,
    batch_indices: Optional[Sequence[int]] = None,
    k: int = 5,
    print_fn: Callable[..., None] = print,
    header: str = "DualXDA τ debug",
) -> None:
    """Pretty-print :func:`format_dualxda_debug_lines` (for scripts / notebooks)."""
    print_fn(header)
    for line in format_dualxda_debug_lines(attr, tri, batch_indices=batch_indices, k=k):
        print_fn(line)


class DualXDATracer:
    """
    Thin wrapper around dualxda.DualDA that turns attributions into axiom signals.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: Any,
        layer_name: str,
        device: str,
        cache_dir: str,
        C: float = 1.0,
        thresholds: Optional[AxiomThresholds] = None,
        intruder_rule: str = "structural",
    ):
        try:
            from dualxda import DualDA  # type: ignore
        except Exception as e:
            # Fallback: this repo sometimes runs with a venv that does not have
            # `dualxda` installed, but the project also contains a local
            # `dualxda-pip/` checkout under `old_pilots/`.
            local_dualxda_pip = Path(
                "/Users/andrearachetta/Documents/old_pilots/dtag_rotber_agent/dualxda-pip"
            )
            if local_dualxda_pip.exists():
                import sys

                # Put the local package ahead of site-packages to ensure we import
                # the correct DualDA implementation.
                sys.path.insert(0, str(local_dualxda_pip))
                from dualxda import DualDA  # type: ignore
            else:
                raise ImportError(
                    "dualxda is required for DualXDA training-trace axioms. "
                    "Either install it with `pip install dualxda` or ensure the local "
                    "checkout exists at: "
                    f"{local_dualxda_pip}"
                ) from e

        self.model = model
        self.train_dataset = train_dataset
        self.layer_name = layer_name
        self.device = device
        self.cache_dir = str(Path(cache_dir))
        self.C = float(C)
        self.thresholds = thresholds or AxiomThresholds(
            tau_mass=0.0, tau_signal=0.04, tau_dom=0.13, pile_metric="coherence"
        )

        # Labels are optional; only needed for "Intruder" flavor diagnostics.
        self.train_labels = _infer_labels_from_dataset(train_dataset)

        # structural: intruder := intruder_candidate (matches axiom text in triage docstring).
        # inverse_mirror: additionally require dominant-trace label conflict + positive dom attribution.
        self.intruder_rule = str(intruder_rule or "structural")

        # DualDA API: dataset, model, classifier_layer, device, cache_dir, C
        self.da = DualDA(
            dataset=train_dataset,
            model=model,
            classifier_layer=layer_name,
            device=device,
            cache_dir=self.cache_dir,
            C=self.C,
        )

    def remove_hook(self) -> None:
        try:
            self.da.remove_hook()
        except Exception:
            pass

    @torch.no_grad()
    def traces(self, x: torch.Tensor, targets: torch.Tensor, drop_zero_columns: bool = False) -> torch.Tensor:
        """
        Returns:
          attr: Tensor [B, N_train] of signed attributions (training traces).
        """
        return self.da.attribute(x=x, xpl_targets=targets, drop_zero_columns=drop_zero_columns)

    @torch.no_grad()
    def axiom_signals(self, attr: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute (mass, signal, dominance, argmax index) per sample.
        """
        return compute_axiom_signals(attr, self.thresholds)

    @torch.no_grad()
    def triage(self, attr: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Apply threshold logic to produce MECE-ish axiom assignments.

        Axiom III (Void):
          mass < tau_mass

        Axiom I (Pile):
          mass >= tau_mass  AND  pile_score <= tau_signal
          where pile_score is either |sum T| (pile_metric=absolute_signal) or
          coherence = |sum T| / (sum|T|+eps) (pile_metric=coherence).

        Axiom II (Intruder):
          mass >= tau_mass  AND  dominance >= tau_dom  AND  pile_score > tau_signal

        Note: With DualDA over the full training set, max|T|/sum|T| is often << 0.8;
        calibrate tau_dom using run diagnostics (typical range ~0.05–0.25 on CIFAR).

        Optional: intruder_rule=inverse_mirror tightens Intruder using label conflict on
        the dominant trace (requires train labels).
        """
        return compute_triage(
            attr,
            self.thresholds,
            targets=targets,
            train_labels=self.train_labels,
            intruder_rule=self.intruder_rule,
        )

    @torch.no_grad()
    def uncertainty_scores(self, attr: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute 3 DualXDA-based uncertainty scores for AURC analysis.
        
        Returns:
            Dictionary with:
              - signal_uncertainty: Inverted absolute signal (high = conflicting influences)
              - triage_uncertainty: Axiom-based uncertainty levels (void=1.0, pile=0.6, intruder=0.7)
              - spurious_score: Dominance × signal_weakness (high = spurious features)
              - Plus diagnostic components: mass, signal, coherence, dominance
        """
        return compute_axiom_uncertainty_scores(attr, self.thresholds)

    @torch.no_grad()
    def debug_print_batch(
        self,
        attr: torch.Tensor,
        tri: Optional[Dict[str, torch.Tensor]] = None,
        *,
        targets: Optional[torch.Tensor] = None,
        batch_indices: Optional[Sequence[int]] = None,
        k: int = 5,
    ) -> None:
        """
        Print mass / coherence / dominance / axiom and top-``k`` positive & negative ``T_i``
        for selected batch rows. Pass ``tri`` from ``self.triage(attr, targets=...)``, or
        omit ``tri`` and pass ``targets`` to triage internally.
        """
        if tri is None:
            if targets is None:
                raise ValueError("debug_print_batch requires tri= or targets=")
            tri = self.triage(attr, targets=targets)
        print_dualxda_batch_debug(
            attr,
            tri,
            batch_indices=batch_indices,
            k=k,
            header="DualXDATracer τ debug",
        )
