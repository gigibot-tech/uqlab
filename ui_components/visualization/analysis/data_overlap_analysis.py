"""
Data Overlap Analysis UI Component

Analyzes and visualizes data overlap between experiments,
including class distributions and index overlaps.
"""

import streamlit as st
import torch
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import Counter
import numpy as np


def load_experiment_data(experiment_id: str) -> Dict:
    """Load training and evaluation data from results.pt"""
    results_path = Path(f"/tmp/uqlab_experiments/{experiment_id}/results/results.pt")
    
    if not results_path.exists():
        return None
    
    try:
        data = torch.load(results_path, map_location='cpu', weights_only=False)
        return {
            'train_indices': data.get('train_indices'),
            'train_labels': data.get('train_labels'),
            'train_is_noisy': data.get('train_is_noisy'),
            'eval_indices': data.get('eval_indices'),
            'eval_labels': data.get('eval_clean_labels'),
            'eval_group_labels': data.get('eval_group_labels'),
        }
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None


def calculate_class_distribution(labels: torch.Tensor) -> Dict[int, int]:
    """Calculate class distribution from labels"""
    if labels is None:
        return {}
    
    labels_np = labels.numpy() if isinstance(labels, torch.Tensor) else labels
    return dict(Counter(labels_np.tolist()))


def calculate_overlap(indices_a: Set[int], indices_b: Set[int]) -> Tuple[int, float]:
    """Calculate overlap between two sets of indices"""
    overlap = len(indices_a & indices_b)
    total = len(indices_a | indices_b)
    overlap_pct = (overlap / total * 100) if total > 0 else 0
    return overlap, overlap_pct


def find_similar_experiments(
    target_exp_id: str,
    target_data: Dict,
    all_experiments: List[Dict]
) -> List[Tuple[str, str, float, Dict]]:
    """
    Find experiments with similar data composition.
    
    Returns list of (exp_id, exp_name, similarity_score, overlap_details)
    """
    if target_data is None or target_data['train_indices'] is None:
        return []
    
    target_train_set = set(target_data['train_indices'].numpy().tolist())
    target_eval_set = set(target_data['eval_indices'].numpy().tolist()) if target_data['eval_indices'] is not None else set()
    target_class_dist = calculate_class_distribution(target_data['train_labels'])
    
    similarities = []
    
    for exp in all_experiments:
        if exp['id'] == target_exp_id or exp.get('status') != 'completed':
            continue
        
        exp_data = load_experiment_data(exp['id'])
        if exp_data is None or exp_data['train_indices'] is None:
            continue
        
        exp_train_set = set(exp_data['train_indices'].numpy().tolist())
        exp_eval_set = set(exp_data['eval_indices'].numpy().tolist()) if exp_data['eval_indices'] is not None else set()
        exp_class_dist = calculate_class_distribution(exp_data['train_labels'])
        
        # Calculate overlaps
        train_overlap, train_overlap_pct = calculate_overlap(target_train_set, exp_train_set)
        eval_overlap, eval_overlap_pct = calculate_overlap(target_eval_set, exp_eval_set)
        
        # Calculate class distribution similarity (cosine similarity)
        all_classes = set(target_class_dist.keys()) | set(exp_class_dist.keys())
        vec_a = np.array([target_class_dist.get(c, 0) for c in sorted(all_classes)])
        vec_b = np.array([exp_class_dist.get(c, 0) for c in sorted(all_classes)])
        
        class_similarity = np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b) + 1e-8)
        
        # Combined similarity score
        similarity_score = (train_overlap_pct + eval_overlap_pct + class_similarity * 100) / 3
        
        overlap_details = {
            'train_overlap': train_overlap,
            'train_overlap_pct': train_overlap_pct,
            'eval_overlap': eval_overlap,
            'eval_overlap_pct': eval_overlap_pct,
            'class_similarity': class_similarity,
            'class_dist': exp_class_dist,
        }
        
        similarities.append((exp['id'], exp['name'], similarity_score, overlap_details))
    
    # Sort by similarity score (descending)
    similarities.sort(key=lambda x: x[2], reverse=True)
    
    return similarities[:3]  # Top 3


def render_data_overlap_analysis(
    selected_exp: Dict,
    all_experiments: List[Dict]
):
    """
    Render comprehensive data overlap analysis.
    
    Args:
        selected_exp: Selected experiment dict
        all_experiments: List of all experiments
    """
    st.markdown("### 🔍 Data Overlap Analysis")
    
    exp_id = selected_exp['id']
    exp_data = load_experiment_data(exp_id)
    
    if exp_data is None:
        st.warning("⚠️ Could not load experiment data for analysis")
        return
    
    # ========== SECTION 1: Current Experiment Data Summary ==========
    st.markdown("#### 📊 Current Experiment Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Training Data:**")
        if exp_data['train_indices'] is not None:
            train_indices = exp_data['train_indices'].numpy().tolist()
            st.text(f"Training samples: {len(train_indices)}")
            st.text(f"Dataset indices: {train_indices[:10]}...")
            
            if exp_data['train_is_noisy'] is not None:
                clean_count = (~exp_data['train_is_noisy']).sum().item()
                noisy_count = exp_data['train_is_noisy'].sum().item()
                st.text(f"Clean samples: {clean_count}")
                st.text(f"Noisy samples: {noisy_count}")
    
    with col2:
        st.markdown("**Evaluation Data:**")
        if exp_data['eval_indices'] is not None:
            eval_indices = exp_data['eval_indices'].numpy().tolist()
            st.text(f"Evaluation samples: {len(eval_indices)}")
            st.text(f"Dataset indices: {eval_indices[:10]}...")
            
            if exp_data['eval_group_labels'] is not None:
                group_counts = Counter(exp_data['eval_group_labels'].numpy().tolist())
                st.text(f"Clean: {group_counts.get(0, 0)}")
                st.text(f"Aleatoric: {group_counts.get(1, 0)}")
                st.text(f"Epistemic: {group_counts.get(2, 0)}")
    
    # ========== SECTION 2: Class Distribution ==========
    st.markdown("---")
    st.markdown("#### 📈 Class Distribution")
    
    class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    
    train_class_dist = calculate_class_distribution(exp_data['train_labels'])
    eval_class_dist = calculate_class_distribution(exp_data['eval_labels'])
    
    # Create distribution dataframe
    dist_data = []
    for class_id in range(10):
        dist_data.append({
            'Class': f"{class_id}: {class_names[class_id]}",
            'Training': train_class_dist.get(class_id, 0),
            'Evaluation': eval_class_dist.get(class_id, 0),
        })
    
    dist_df = pd.DataFrame(dist_data)
    st.dataframe(dist_df, use_container_width=True, hide_index=True)
    
    # ========== SECTION 3: Similar Experiments ==========
    st.markdown("---")
    st.markdown("#### 🔗 Most Similar Experiments")
    st.caption("Based on data overlap and class distribution similarity")
    
    with st.spinner("Analyzing all experiments..."):
        similar_exps = find_similar_experiments(exp_id, exp_data, all_experiments)
    
    if not similar_exps:
        st.info("No similar experiments found")
        return
    
    for rank, (sim_id, sim_name, similarity_score, overlap_details) in enumerate(similar_exps, 1):
        with st.expander(f"#{rank}: {sim_name} ({sim_id[:8]}) - Similarity: {similarity_score:.1f}%", expanded=(rank == 1)):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Training Overlap:**")
                st.metric(
                    "Shared Samples",
                    overlap_details['train_overlap'],
                    f"{overlap_details['train_overlap_pct']:.1f}%"
                )
            
            with col2:
                st.markdown("**Evaluation Overlap:**")
                st.metric(
                    "Shared Samples",
                    overlap_details['eval_overlap'],
                    f"{overlap_details['eval_overlap_pct']:.1f}%"
                )
            
            with col3:
                st.markdown("**Class Similarity:**")
                st.metric(
                    "Distribution Match",
                    f"{overlap_details['class_similarity']:.3f}",
                    f"{overlap_details['class_similarity'] * 100:.1f}%"
                )
            
            # Show class distribution comparison
            st.markdown("**Class Distribution Comparison:**")
            comp_data = []
            for class_id in range(10):
                comp_data.append({
                    'Class': f"{class_id}: {class_names[class_id]}",
                    'Current': train_class_dist.get(class_id, 0),
                    'Similar': overlap_details['class_dist'].get(class_id, 0),
                    'Difference': abs(train_class_dist.get(class_id, 0) - overlap_details['class_dist'].get(class_id, 0)),
                })
            
            comp_df = pd.DataFrame(comp_data)
            st.dataframe(comp_df, use_container_width=True, hide_index=True)


# Made with Bob