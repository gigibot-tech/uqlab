"""Shim: ``uq_classification.evaluation`` → ``uqlab.4_evaluation.evaluator``."""

import importlib

_evaluator = importlib.import_module("uqlab.4_evaluation.evaluator")

# Re-export public API used by scripts and Streamlit.
evaluate_three_way_classification = _evaluator.evaluate_three_way_classification
predict_eval_groups_single_signal = _evaluator.predict_eval_groups_single_signal
binary_auroc = _evaluator.binary_auroc
split_group_balanced_targets = _evaluator.split_group_balanced_targets
save_per_sample_csv = _evaluator.save_per_sample_csv
print_noisy_eval_samples = _evaluator.print_noisy_eval_samples
build_results_markdown = _evaluator.build_results_markdown
train_signal_classifier = _evaluator.train_signal_classifier

__all__ = [
    "binary_auroc",
    "build_results_markdown",
    "evaluate_three_way_classification",
    "predict_eval_groups_single_signal",
    "print_noisy_eval_samples",
    "save_per_sample_csv",
    "split_group_balanced_targets",
    "train_signal_classifier",
]
