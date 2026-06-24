# Upstream: disentanglement_error

Vendored from [ivopascal/disentanglement_error](https://github.com/ivopascal/disentanglement_error).

Paper: *Measuring the Disentanglement of Aleatoric and Epistemic Uncertainty in Deep Neural Networks* (Pascal et al.).

Local changes:

- Imports use `uqlab.vendor.disentanglement_error.*`
- `label_noise_experiment` / `decreasing_dataset_experiment` pass sweep kwargs to `model.fit`
- `util.AverageMeter.all_reduce` lazy-imports `torch.distributed`
- `error_metric.calculate_*` treats `kw_config=None` as empty `Config()`
