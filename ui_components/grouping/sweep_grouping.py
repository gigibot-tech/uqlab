"""
Intelligent Sweep Grouping Utilities

This module provides utilities for detecting and grouping parameter sweeps
from individual experiments by analyzing config similarity.

Supports both:
- Option 1: Explicit sweep metadata (sweep_group_id, swept_parameter, etc.)
- Option 2: Config-based detection (analyzes YAML configs for single-parameter differences)
"""

from typing import Dict, List, Tuple, Any, Optional
import yaml
from datetime import datetime
import streamlit as st


def _flatten_dict(d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
    """
    Flatten nested dictionary for easier comparison.
    
    Example:
        {'model': {'hidden_dim': 256}} -> {'model.hidden_dim': 256}
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _find_config_differences(config1: Dict, config2: Dict) -> List[Tuple[str, Any, Any]]:
    """
    Find differences between two configs.
    
    Returns:
        List of (key, value1, value2) tuples for differing parameters
    """
    flat1 = _flatten_dict(config1)
    flat2 = _flatten_dict(config2)
    
    differences = []
    
    # Check all keys in both configs
    all_keys = set(flat1.keys()) | set(flat2.keys())
    
    for key in all_keys:
        val1 = flat1.get(key)
        val2 = flat2.get(key)
        
        if val1 != val2:
            differences.append((key, val1, val2))
    
    return differences


def _extract_swept_value(exp: Dict, param_key: str) -> Optional[Any]:
    """
    Extract the value of a swept parameter from an experiment's config.
    
    Args:
        exp: Experiment dict with 'config_yaml' field
        param_key: Flattened parameter key (e.g., 'evaluation.mc_passes')
    
    Returns:
        The parameter value, or None if not found
    """
    try:
        config = yaml.safe_load(exp['config_yaml'])
        flat_config = _flatten_dict(config)
        return flat_config.get(param_key)
    except:
        return None


def group_experiments_by_config_similarity(
    experiments: List[Dict],
    min_group_size: int = 3
) -> Tuple[List[Dict], List[Dict]]:
    """
    Group experiments by config similarity to detect script-generated sweeps.
    
    Algorithm:
    1. Parse all experiment configs
    2. For each experiment, find others that differ by exactly ONE parameter
    3. Group experiments with the same swept parameter
    4. Return groups with >= min_group_size experiments
    
    Args:
        experiments: List of experiment dicts with 'config_yaml' field
        min_group_size: Minimum experiments to form a sweep group (default: 3)
    
    Returns:
        (sweep_groups, standalone_experiments)
        
        sweep_groups: List of dicts with:
            - name: str (e.g., "Sweep: mc_passes")
            - swept_param: str (e.g., "evaluation.mc_passes")
            - experiments: List[Dict] (sorted by swept value)
            - values: List[Any] (swept values in order)
            
        standalone_experiments: List of experiments not in any sweep
    """
    # Parse all configs
    configs = []
    for exp in experiments:
        try:
            config = yaml.safe_load(exp['config_yaml'])
            configs.append((exp, config))
        except Exception as e:
            # Skip experiments with invalid configs
            continue
    
    if len(configs) < min_group_size:
        return [], experiments
    
    # Find sweep groups
    sweep_groups = []
    used_exp_ids = set()
    
    for i, (exp1, config1) in enumerate(configs):
        if exp1['id'] in used_exp_ids:
            continue
        
        # Try to build a group starting from this experiment
        group = [exp1]
        swept_param = None
        
        for j in range(i + 1, len(configs)):
            exp2, config2 = configs[j]
            
            if exp2['id'] in used_exp_ids:
                continue
            
            # Find differences between configs
            diffs = _find_config_differences(config1, config2)
            
            # Only group if exactly ONE parameter differs
            if len(diffs) == 1:
                diff_key, val1, val2 = diffs[0]
                
                if swept_param is None:
                    # First match - establish the swept parameter
                    swept_param = diff_key
                    group.append(exp2)
                    used_exp_ids.add(exp2['id'])
                elif swept_param == diff_key:
                    # Same parameter - add to group
                    group.append(exp2)
                    used_exp_ids.add(exp2['id'])
        
        # Only create a group if we have enough experiments
        if len(group) >= min_group_size and swept_param:
            # Extract swept values and sort
            group_with_values = []
            for exp in group:
                value = _extract_swept_value(exp, swept_param)
                group_with_values.append((exp, value))
            
            # Sort by swept value (handle None values)
            group_with_values.sort(key=lambda x: (x[1] is None, x[1]))
            
            sweep_groups.append({
                'name': f"Sweep: {swept_param.split('.')[-1]}",  # Use last part of key
                'swept_param': swept_param,
                'experiments': [exp for exp, _ in group_with_values],
                'values': [val for _, val in group_with_values],
                'created_at': min(exp['created_at'] for exp, _ in group_with_values)
            })
            used_exp_ids.add(exp1['id'])
    
    # Remaining experiments are standalone
    standalone = [exp for exp in experiments if exp['id'] not in used_exp_ids]
    
    return sweep_groups, standalone


def group_experiments_by_metadata(
    experiments: List[Dict]
) -> Tuple[List[Dict], List[Dict]]:
    """
    Group experiments by explicit sweep metadata (Option 1).
    
    Looks for:
    - sweep_group_id: Unique identifier for the sweep
    - swept_parameter: Name of the parameter being swept
    - swept_value: Value of the parameter for this experiment
    - sweep_index: Position in the sweep (for sorting)
    
    Args:
        experiments: List of experiment dicts
    
    Returns:
        (sweep_groups, standalone_experiments)
    """
    sweep_groups_by_id = {}
    standalone = []
    
    for exp in experiments:
        sweep_id = exp.get('sweep_group_id')
        
        if sweep_id:
            if sweep_id not in sweep_groups_by_id:
                sweep_groups_by_id[sweep_id] = []
            sweep_groups_by_id[sweep_id].append(exp)
        else:
            standalone.append(exp)
    
    # Convert to sweep group format
    sweep_groups = []
    for sweep_id, exps in sweep_groups_by_id.items():
        # Sort by sweep_index if available
        exps_sorted = sorted(exps, key=lambda e: e.get('sweep_index', 0))
        
        sweep_groups.append({
            'name': f"Sweep: {exps_sorted[0].get('swept_parameter', 'unknown')}",
            'swept_param': exps_sorted[0].get('swept_parameter', 'unknown'),
            'experiments': exps_sorted,
            'values': [e.get('swept_value') for e in exps_sorted],
            'sweep_group_id': sweep_id,
            'created_at': min(e['created_at'] for e in exps_sorted)
        })
    
    return sweep_groups, standalone


def group_experiments_by_name_pattern(
    experiments: List[Dict],
    min_group_size: int = 3
) -> Tuple[List[Dict], List[Dict]]:
    """
    Group experiments by name pattern to detect script-generated sweeps.
    
    Detects patterns like:
    - fast_alea_20260615_174459_noise_100
    - fast_alea_20260615_174459_noise_75
    - fast_alea_20260615_174459_noise_50
    
    Pattern: {prefix}_{timestamp}_{param}_{value}
    Where prefix and timestamp are the same across the sweep.
    
    Algorithm:
    1. Split experiment names by underscore
    2. Extract prefix (index 0) and timestamp (indices 1-2)
    3. Group experiments with same prefix+timestamp
    4. Detect swept parameter from the varying suffix
    
    Args:
        experiments: List of experiment dicts with 'name' field
        min_group_size: Minimum experiments to form a sweep group (default: 3)
    
    Returns:
        (sweep_groups, standalone_experiments)
    """
    from collections import defaultdict
    
    # Group by (prefix, timestamp)
    groups_by_pattern = defaultdict(list)
    
    for exp in experiments:
        name = exp.get('name', '')
        parts = name.split('_')
        
        # Need at least 4 parts: prefix_date_time_param_value
        if len(parts) >= 4:
            # Extract prefix (index 0) and timestamp (indices 1-2)
            prefix = parts[0]
            timestamp = f"{parts[1]}_{parts[2]}"  # e.g., "20260615_174459"
            pattern_key = f"{prefix}_{timestamp}"
            
            groups_by_pattern[pattern_key].append(exp)
        else:
            # Can't parse pattern, treat as standalone
            groups_by_pattern[name].append(exp)
    
    # Convert to sweep groups
    sweep_groups = []
    standalone = []
    
    for pattern_key, exps in groups_by_pattern.items():
        if len(exps) >= min_group_size:
            # This is a sweep group
            # Try to detect swept parameter from name suffixes
            swept_param = "unknown"
            values = []
            
            for exp in exps:
                name = exp.get('name', '')
                parts = name.split('_')
                
                # Extract parameter and value from suffix
                # Pattern: prefix_date_time_param_value
                if len(parts) >= 5:
                    # Last part is value, second-to-last is parameter
                    param = parts[-2]
                    value = parts[-1]
                    
                    if swept_param == "unknown":
                        swept_param = param
                    
                    values.append(value)
                else:
                    values.append("unknown")
            
            # Sort experiments by swept value (try numeric first, fall back to string)
            try:
                # Try to sort numerically
                exps_with_values = list(zip(exps, values))
                exps_with_values.sort(key=lambda x: float(x[1]) if x[1].replace('.', '').replace('-', '').isdigit() else x[1])
                exps_sorted = [e for e, v in exps_with_values]
                values_sorted = [v for e, v in exps_with_values]
            except:
                # Fall back to string sorting
                exps_with_values = list(zip(exps, values))
                exps_with_values.sort(key=lambda x: x[1])
                exps_sorted = [e for e, v in exps_with_values]
                values_sorted = [v for e, v in exps_with_values]
            
            sweep_groups.append({
                'name': f"Sweep: {swept_param} (from name pattern)",
                'swept_param': swept_param,
                'experiments': exps_sorted,
                'values': values_sorted,
                'sweep_group_id': pattern_key,
                'created_at': min(e['created_at'] for e in exps_sorted),
                'source': 'name_pattern'
            })
        else:
            # Not enough experiments, treat as standalone
            standalone.extend(exps)
    
    # Sort groups by creation time (newest first)
    sweep_groups.sort(key=lambda g: g['created_at'], reverse=True)
    
    return sweep_groups, standalone


def group_experiments_intelligently(
    experiments: List[Dict],
    min_group_size: int = 3
) -> Tuple[List[Dict], List[Dict]]:
    """
    Hybrid approach: Try multiple grouping strategies in order of reliability.
    
    Strategy priority:
    1. Metadata-based (Option 1) - explicit sweep_group_id
    2. Name pattern-based - detects prefix_timestamp_param_value patterns
    3. Config-based (Option 2) - analyzes YAML config differences
    
    This is the recommended entry point for sweep grouping.
    
    Args:
        experiments: List of experiment dicts
        min_group_size: Minimum experiments to form a sweep group (default: 3)
    
    Returns:
        (sweep_groups, standalone_experiments)
    """
    all_groups = []
    
    # Strategy 1: Try metadata-based grouping (Option 1)
    metadata_groups, remaining = group_experiments_by_metadata(experiments)
    all_groups.extend(metadata_groups)
    
    # Strategy 2: Try name pattern-based grouping
    if len(remaining) >= min_group_size:
        name_groups, remaining = group_experiments_by_name_pattern(
            remaining,
            min_group_size=min_group_size
        )
        all_groups.extend(name_groups)
    
    # Strategy 3: Try config-based detection (Option 2)
    if len(remaining) >= min_group_size:
        config_groups, remaining = group_experiments_by_config_similarity(
            remaining,
            min_group_size=min_group_size
        )
        all_groups.extend(config_groups)
    
    # Sort all groups by creation time (newest first)
    all_groups.sort(key=lambda g: g['created_at'], reverse=True)
    
    return all_groups, remaining


def render_sweep_group_summary(group: Dict, show_details: bool = False, api_base_url: Optional[str] = None):
    """
    Render a summary card for a sweep group.
    
    Args:
        group: Sweep group dict from group_experiments_intelligently()
        show_details: Whether to show detailed experiment list
        api_base_url: Base URL for API (for fetching detailed metrics)
    """
    experiments = group['experiments']
    swept_param = group['swept_param']
    values = group['values']
    
    # Calculate summary statistics
    completed = sum(1 for e in experiments if e['status'] == 'completed')
    running = sum(1 for e in experiments if e['status'] == 'running')
    failed = sum(1 for e in experiments if e['status'] == 'failed')
    
    # Get AUROC scores for completed experiments
    aleatoric_scores = [e.get('aleatoric_auroc') for e in experiments if e.get('aleatoric_auroc')]
    epistemic_scores = [e.get('epistemic_auroc') for e in experiments if e.get('epistemic_auroc')]
    
    # Display summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Runs", len(experiments))
    with col2:
        st.metric("Completed", f"{completed}/{len(experiments)}")
    with col3:
        if aleatoric_scores:
            best_aleatoric = max(aleatoric_scores)
            st.metric("Best Aleatoric AUROC", f"{best_aleatoric:.3f}")
        else:
            st.metric("Best Aleatoric AUROC", "N/A")
    with col4:
        if epistemic_scores:
            best_epistemic = max(epistemic_scores)
            st.metric("Best Epistemic AUROC", f"{best_epistemic:.3f}")
        else:
            st.metric("Best Epistemic AUROC", "N/A")
    
    # Show swept values
    st.markdown(f"**Swept Parameter:** `{swept_param}`")
    st.markdown(f"**Values:** {', '.join(str(v) for v in values)}")
    
    # Show status breakdown
    status_text = f"✅ {completed} completed"
    if running > 0:
        status_text += f" | 🔄 {running} running"
    if failed > 0:
        status_text += f" | ❌ {failed} failed"
    st.caption(status_text)
    
    if show_details:
        st.markdown("---")
        st.markdown("**Experiment Details:**")
        
        # Create table with View Details buttons
        import pandas as pd
        
        data = []
        for exp, value in zip(experiments, values):
            data.append({
                'Value': value,
                'Name': exp['name'],
                'Status': exp['status'],
                'Aleatoric': f"{exp.get('aleatoric_auroc', 0):.3f}" if exp.get('aleatoric_auroc') else "N/A",
                'Epistemic': f"{exp.get('epistemic_auroc', 0):.3f}" if exp.get('epistemic_auroc') else "N/A",
            })
        
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Add "View Details" buttons for completed experiments
        st.markdown("**📊 View Detailed Metrics:**")
        completed_exps = [e for e in experiments if e['status'] == 'completed' and e.get('best_signals_json')]
        
        if completed_exps:
            for exp in completed_exps:
                exp_id = exp['id']
                exp_name = exp['name']
                
                # Use expander for each experiment's detailed view
                with st.expander(f"🔬 {exp_name} - Detailed Metrics", expanded=False):
                    # Import here to avoid circular dependency
                    from uqlab.ui_components.results.experiment_details import render_experiment_details_with_metrics
                    render_experiment_details_with_metrics(exp, show_explanation=False)
        else:
            st.info("💡 Detailed metrics will be available after experiments complete")

# Made with Bob
