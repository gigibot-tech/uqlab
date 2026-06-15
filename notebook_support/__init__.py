"""
Backward compatibility for ``uqlab.notebook_support.<submodule>``.

Import submodules directly (e.g. ``metric_specs``, ``signals``) to avoid loading
the full ``uqlab.shared`` package tree.
"""

__all__: list[str] = []
