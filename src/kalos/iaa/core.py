"""
The mathematical engine for KaLOS Inter-Annotator Agreement (IAA).
Implements the "Vision Alpha" variant of Krippendorff's Alpha, alongside
diagnostic metrics for Annotator Vitality and Category Difficulty.
"""

import itertools
import logging
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from kalos.correspondence import correspondence_algorithms
from tqdm import tqdm

logger = logging.getLogger(__name__)

def calculate_iaa(
    processed_data: Dict[int, Dict[str, Any]],
    categories: Dict[int, str],
    method: str,
    threshold_func: str,
    cost_func: str,
    similarity_threshold: float,
    calculate_vitality: bool,
    calculate_difficulty: bool,
    collaboration_clusters: bool,
    all_raters: List[str],
) -> Tuple[float, float, Dict, Dict, Dict, Dict, Dict, Dict, Dict]:
    """
    Orchestrates the calculation of Inter-Annotator Agreement (IAA).

    Calculates both Mean-Image Alpha (Primary) and Global Dataset-Wide Alpha (Secondary).
    Optionally computes diagnostic metrics for rater influence and class difficulty.

    Args:
        processed_data (Dict[int, Dict[str, Any]]): Preprocessed annotation data grouped by image.
        categories (Dict[int, str]): Map of category IDs to names.
        method (str): Matching algorithm name (e.g., 'greedy', 'mgm').
        threshold_func (str): Similarity function name (e.g., 'segm_iou_similarity').
        cost_func (str): Cost function name (e.g., 'category_lenient').
        similarity_threshold (float): Localization threshold for matching.
        calculate_vitality (bool): Whether to calculate rater sensitivity (Vitality).
        calculate_difficulty (bool): Whether to calculate category-wise agreement.
        collaboration_clusters (bool): Whether to calculate pairwise rater agreement matrices.
        all_raters (List[str]): Full list of rater identities across the dataset.

    Returns:
        Tuple containing:
            - mean_alpha (float): Average Alpha across all images.
            - global_alpha (float): Dataset-wide Alpha calculated from the global reliability matrix.
            - mean_vitalities (Dict): Map of rater IDs to their average vitality scores.
            - global_vitalities (Dict): Map of rater IDs to their global vitality scores.
            - mean_difficulties (Dict): Map of class names to their mean agreement and vitality stats.
            - global_difficulties (Dict): Map of class names to their global agreement and vitality stats.
            - image_alphas (Dict): Map of image IDs to their individual Alpha scores.
            - mean_collaboration_matrix (Dict): Nested map of pairwise rater agreement scores (lists of per-image alphas).
            - global_collaboration_matrix (Dict): Nested map of global pairwise rater agreement scores (single values).
    """
    # Safeguard: -1.0 is reserved for existence disagreement (missing object)
    if -1 in categories or -1.0 in categories:
        raise ValueError("Category ID -1.0 is reserved in KaLOS for 'existence disagreement'. Please use non-negative integers for categories.")

    threshold_function = correspondence_algorithms.THRESHOLD_FUNCTIONS[threshold_func]
    cost_function = correspondence_algorithms.COST_FUNCTIONS[cost_func]
    matching_function = correspondence_algorithms.MATCHING_FUNCTIONS[method]

    image_alphas = {}
    mean_vitalities = defaultdict(list)
    # alphas for all classes and alpha_ki for the vitality change for this specific class
    mean_difficulties = defaultdict(lambda: {"alphas": [], "alpha_ki": defaultdict(list)})
    mean_collaboration_matrix = defaultdict(lambda: defaultdict(list))
    global_pairwise_units = defaultdict(list)

    # --- Global Matrix Accumulation ---
    all_units = []
    rater_to_idx = {rater_id: i for i, rater_id in enumerate(all_raters)}

    for image_id, image_data in tqdm(processed_data.items(), desc=f"Calculating IAA with threshold {similarity_threshold}", unit="image", leave=False):
        # 1. Pre-compute pairwise scores
        pairwise_scores = correspondence_algorithms.precompute_pairwise_scores(
            image_data, threshold_function, similarity_threshold
        )

        # 2. Calculate overall alpha for the image
        correspondence_clusters = matching_function(
            image_data=image_data,
            pairwise_scores=pairwise_scores,
            cost_func=cost_function,
            similarity_threshold=similarity_threshold
        )
        reliability_data = build_reliability_matrix(image_data, correspondence_clusters)
        image_alpha = vision_alpha(reliability_data)
        image_alphas[image_id] = image_alpha

        # --- Accumulate for Global Alpha ---
        image_raters = set(image_data['rater_list'])
        if not correspondence_clusters:
            # Agreement on empty image: Add one virtual unit
            unit = [np.nan] * len(all_raters)
            for r_id in image_raters:
                unit[rater_to_idx[r_id]] = -1.0
            all_units.append(unit)
        else:
            # Standard units from clusters
            ann_id_to_annotation = {
                ann['id']: ann 
                for rater_anns in image_data['annotations_by_rater'].values() 
                for ann in rater_anns
            }
            for cluster in correspondence_clusters:
                unit = [np.nan] * len(all_raters)
                for r_id in image_raters:
                    unit[rater_to_idx[r_id]] = -1.0 # Default to "Missing object" for assigned raters
                for ann_id in cluster:
                    ann = ann_id_to_annotation.get(ann_id)
                    if ann:
                        unit[rater_to_idx[ann['rater_id']]] = ann['category_id']
                all_units.append(unit)

        # 3. Class Difficulty (Mean accumulation)
        class_difficulty = None
        if calculate_difficulty:
            class_difficulty = calculate_class_difficulty(reliability_data)
            for cls, cls_alpha in class_difficulty.items():
                if cls in categories:
                    mean_difficulties[categories[cls]]["alphas"].append(cls_alpha)

        # 4. Annotator Vitality (Mean accumulation)
        if calculate_vitality:
            annotator_vitality, class_difficulties_ki = calculate_image_rater_vitality(
                image_data,
                pairwise_scores,
                image_alpha,
                class_difficulty,
                cost_func,
                method,
                similarity_threshold,
                calculate_difficulty,
            )
            for rater_id, vitality in annotator_vitality.items():
                mean_vitalities[rater_id].append(vitality)
            if calculate_difficulty:
                for cls, cls_alpha_ki in class_difficulties_ki.items():
                    if cls in categories:
                        for rater_id, alpha_ki in cls_alpha_ki.items():
                            mean_difficulties[categories[cls]]["alpha_ki"][rater_id].append(alpha_ki)

        # 5. Collaboration Clusters
        if collaboration_clusters:
            for rater_1, rater_2 in itertools.combinations(image_data['rater_list'], 2):
                pair_image_data = {
                    'rater_list': [rater_1, rater_2],
                    'annotations_by_rater': {
                        rater_1: image_data['annotations_by_rater'][rater_1],
                        rater_2: image_data['annotations_by_rater'][rater_2]
                    }
                }
                pair_pairwise_scores = correspondence_algorithms.precompute_pairwise_scores(
                    pair_image_data, threshold_function, similarity_threshold
                )
                pair_clusters = matching_function(
                    image_data=pair_image_data,
                    pairwise_scores=pair_pairwise_scores,
                    cost_func=cost_function,
                    similarity_threshold=similarity_threshold
                )
                pair_reliability_data = build_reliability_matrix(pair_image_data, pair_clusters)
                pair_alpha = vision_alpha(pair_reliability_data)
                sorted_pair = tuple(sorted((rater_1, rater_2)))
                mean_collaboration_matrix[sorted_pair[0]][sorted_pair[1]].append(pair_alpha)
                
                # Global Pairwise accumulation
                if not pair_clusters:
                    # Agreement on nothing
                    global_pairwise_units[sorted_pair].append([-1.0, -1.0])
                else:
                    global_pairwise_units[sorted_pair].extend(pair_reliability_data.T.tolist())

    # Final Reductions
    mean_alpha = np.mean(list(image_alphas.values())) if image_alphas else 0.0
    
    global_alpha = 0.0
    global_vitalities = {}
    global_difficulties = {}
    global_collaboration_matrix = defaultdict(dict)

    if all_units:
        global_matrix = np.array(all_units).T
        global_alpha = vision_alpha(global_matrix)

        if calculate_vitality:
            global_vitalities, global_class_vitalities = calculate_global_rater_vitality(
                global_matrix, global_alpha, rater_to_idx, calculate_difficulty
            )

        if calculate_difficulty:
            class_alphas_global = calculate_class_difficulty(global_matrix)
            for cls_id, cls_alpha in class_alphas_global.items():
                if cls_id in categories:
                    cls_name = categories[cls_id]
                    global_difficulties[cls_name] = {
                        "alpha": cls_alpha,
                        "alpha_ki": global_class_vitalities.get(cls_id, {}) if calculate_vitality else {}
                    }
        
        if collaboration_clusters:
            for r1, r2 in itertools.combinations(all_raters, 2):
                sorted_pair = tuple(sorted((r1, r2)))
                units = global_pairwise_units.get(sorted_pair)
                if units:
                    pair_matrix = np.array(units).T
                    global_collaboration_matrix[r1][r2] = vision_alpha(pair_matrix)
                else:
                    global_collaboration_matrix[r1][r2] = np.nan

    return (mean_alpha, global_alpha, mean_vitalities, global_vitalities, 
            mean_difficulties, global_difficulties, image_alphas, 
            mean_collaboration_matrix, global_collaboration_matrix)


def _krippendorff_alpha_nominal(reliability_data: np.ndarray) -> float:
    """
    Memory-efficient implementation of Krippendorff's Alpha for nominal data.
    
    This implementation avoids the O(N * V^2) memory bottleneck of standard 
    implementations by computing the coincidence matrix components using 
    optimized NumPy operations.

    Args:
        reliability_data (np.ndarray): Reliability matrix of shape (num_raters, num_units).
            Expected to contain category IDs as values and np.nan for missing observations.

    Returns:
        float: Calculated Alpha value. Returns 1.0 if agreement is perfect or undefined.
    
    Note:
        Formula: alpha = 1 - (D_o / D_e)
                 alpha = ((n - 1) * sum(o_cc) - sum(n_c * (n_c - 1))) / (n * (n - 1) - sum(n_c * (n_c - 1)))
        Where:
        - n: Total number of paired observations (sum of all coincidences).
        - o_cc: Diagonal elements of the coincidence matrix (observed agreement for category c).
        - n_c: Total count of category c in the coincidence matrix (sum of row/column c).
    """
    # 1. Identify unique values (excluding NaNs)
    mask = ~np.isnan(reliability_data)
    if not np.any(mask):
        return 1.0
    
    unique_vals, inverse = np.unique(reliability_data[mask], return_inverse=True)
    num_vals = len(unique_vals)
    num_units = reliability_data.shape[1]
    
    # 2. Compute value counts per unit (n_ic)
    # counts[i, c] is the number of times unit i was assigned category c
    counts = np.zeros((num_units, num_vals), dtype=int)
    unit_indices = np.where(mask)[1]
    np.add.at(counts, (unit_indices, inverse), 1)
            
    # 3. Filter units with less than 2 observations (m_i < 2)
    m_i = counts.sum(axis=1)
    valid_units = m_i > 1
    if not np.any(valid_units):
        return 1.0
        
    counts = counts[valid_units].astype(float)
    m_i = m_i[valid_units].astype(float)
    
    # 4. Compute components for the formula
    # n = sum of all coincidences = sum(m_i)
    n = np.sum(m_i)
    
    # n_c = sum of row/column c in coincidence matrix = total count of category c
    n_c = counts.sum(axis=0)
    
    # o_cc = diagonal of coincidence matrix = sum_i [ n_ic * (n_ic - 1) / (m_i - 1) ]
    o_cc = np.sum(counts * (counts - 1) / (m_i[:, np.newaxis] - 1), axis=0)
    
    # 5. Calculate Alpha
    if n <= 1:
        return 1.0
        
    sum_occ = np.sum(o_cc)
    sum_nc_nc_minus_1 = np.sum(n_c * (n_c - 1))
    
    numerator = (n - 1) * sum_occ - sum_nc_nc_minus_1
    denominator = n * (n - 1) - sum_nc_nc_minus_1
    
    if denominator == 0:
        return 1.0
        
    return float(numerator / denominator)

def vision_alpha(reliability_data: np.ndarray) -> float:
    """
    Calculates Krippendorff's Alpha with safeguards for Vision-specific edge cases.

    Handles empty matrices, instances with zero overlap, and uniform assignments 
    (where all raters agree perfectly on a single category).

    Args:
        reliability_data (np.ndarray): Reliability matrix of shape (num_raters, num_units).

    Returns:
        float: Alpha score in range [-1.0, 1.0].
    """
    if reliability_data.size == 0:
        return 1.0
    # Check if all non-NaN values are the same
    non_nan_mask = ~np.isnan(reliability_data)
    if not np.any(non_nan_mask):
        return 1.0
    
    first_val = reliability_data[non_nan_mask][0]
    if np.all(reliability_data[non_nan_mask] == first_val):
        return 1.0
        
    return _krippendorff_alpha_nominal(reliability_data)

def calculate_global_rater_vitality(
    global_matrix: np.ndarray,
    global_alpha: float,
    rater_to_idx: Dict[str, int],
    calculate_difficulty: bool,
) -> Tuple[Dict[str, float], Dict[int, Dict[str, float]]]:
    """
    Calculates Global Annotator Vitality via Jackknife-style sensitivity on the global matrix.

    Args:
        global_matrix (np.ndarray): Full reliability matrix.
        global_alpha (float): Baseline global Alpha.
        rater_to_idx (Dict[str, int]): Mapping of rater IDs to matrix row indices.
        calculate_difficulty (bool): Whether to calculate class-level vitality.

    Returns:
        Tuple of (global_vitalities, global_class_vitalities).
    """
    global_vitalities = {}
    global_class_vitalities = defaultdict(dict)
    
    # Baseline class difficulty for Global
    class_difficulty_base = None
    if calculate_difficulty:
        class_difficulty_base = calculate_class_difficulty(global_matrix)

    for rater_id, rater_idx in rater_to_idx.items():
        # Mask out this rater
        mask = np.ones(global_matrix.shape[0], dtype=bool)
        mask[rater_idx] = False
        matrix_ki = global_matrix[mask, :]
        
        # Check if we still have at least 2 raters
        if matrix_ki.shape[0] >= 2:
            ki_alpha = vision_alpha(matrix_ki)
            global_vitalities[rater_id] = global_alpha - ki_alpha
            
            if calculate_difficulty:
                class_difficulty_ki = calculate_class_difficulty(matrix_ki)
                for cls_id, cls_alpha_ki in class_difficulty_ki.items():
                    if cls_id in class_difficulty_base:
                        global_class_vitalities[cls_id][rater_id] = class_difficulty_base[cls_id] - cls_alpha_ki
                        
    return global_vitalities, global_class_vitalities

def calculate_image_rater_vitality(
    image_data: Dict[str, Any],
    pairwise_scores: Dict[Tuple[int, int], Tuple[float, Dict, Dict]],
    image_alpha: float,
    class_difficulty_base: Optional[Dict[int, float]],
    cost_func: str,
    method: str,
    similarity_threshold: float,
    calculate_difficulty: bool,
) -> Tuple[Dict[str, float], Dict[int, Dict[str, float]]]:
    """
    Calculates Annotator Vitality via a Jackknife-style sensitivity analysis.

    Measures the influence of each rater on the image-level agreement by 
    recalculating Alpha with one rater excluded.

    Args:
        image_data (Dict[str, Any]): Preprocessed data for a single image.
        pairwise_scores (Dict): Precomputed similarity scores for annotation pairs.
        image_alpha (float): The original Alpha score for the image.
        class_difficulty_base (Optional[Dict]): Baseline class difficulty scores.
        cost_func (str): Cost function name.
        method (str): Matching algorithm name.
        similarity_threshold (float): Localization threshold.
        calculate_difficulty (bool): Whether to propagate vitality to class level.

    Returns:
        Tuple containing:
            - annotator_vitality (Dict[str, float]): Map of rater IDs to their vitality score.
            - class_difficulties_ki (Dict): Map of class vitality changes per rater.
    """
    cost_function = correspondence_algorithms.COST_FUNCTIONS[cost_func]
    matching_function = correspondence_algorithms.MATCHING_FUNCTIONS[method]

    original_rater_list = image_data['rater_list']
    annotator_vitality = dict()
    class_difficulties_ki = defaultdict(dict)
    
    if len(original_rater_list) >= 3:
        for rater_id_to_exclude in original_rater_list:
            image_data_ki = {
                'rater_list': [r for r in original_rater_list if r != rater_id_to_exclude],
                'annotations_by_rater': {
                    rater_id: anns for rater_id, anns in image_data['annotations_by_rater'].items()
                    if rater_id != rater_id_to_exclude
                }
            }
            allowed_ids_ki = {ann['id'] for anns in image_data_ki['annotations_by_rater'].values() for ann in anns}
            pairwise_scores_ki = {
                pair: values for pair, values in pairwise_scores.items()
                if pair[0] in allowed_ids_ki and pair[1] in allowed_ids_ki
            }
            correspondence_clusters_ki = matching_function(
                image_data=image_data_ki,
                pairwise_scores=pairwise_scores_ki,
                cost_func=cost_function,
                similarity_threshold=similarity_threshold
            )
            reliability_data_ki = build_reliability_matrix(image_data_ki, correspondence_clusters_ki)

            if calculate_difficulty:
                class_difficulty = calculate_class_difficulty(reliability_data_ki)
                for cls, cls_alpha in class_difficulty.items():
                    if class_difficulty_base and cls in class_difficulty_base:
                        class_difficulties_ki[cls][rater_id_to_exclude] = class_difficulty_base[cls] - cls_alpha

            ki = vision_alpha(reliability_data_ki)
            annotator_vitality[rater_id_to_exclude] = image_alpha - ki
            
    return annotator_vitality, class_difficulties_ki


def calculate_class_difficulty(reliability_data: np.ndarray) -> Dict[int, float]:
    """
    Calculates category-specific agreement scores.

    Identifies the difficulty of recognizing specific categories by isolating 
    them within the reliability matrix.

    Args:
        reliability_data (np.ndarray): Reliability matrix of shape (num_raters, num_units).

    Returns:
        Dict[int, float]: Map of category IDs to their localized Alpha scores.
    """
    class_difficulty = {}
    classes = np.unique(reliability_data[~np.isnan(reliability_data)])
    for cls in classes:
        # -1.0 is reserved for no object placeholder (active disagreement); skip it.
        if cls == -1.0: continue
        cols_with_cls = np.any(reliability_data == cls, axis=0)
        cls_alpha = vision_alpha(reliability_data[:, cols_with_cls])
        class_difficulty[cls] = cls_alpha
    return class_difficulty

def build_reliability_matrix(
    image_data: Dict[str, Any], 
    correspondence_clusters: List[Tuple[int, ...]]
) -> np.ndarray:
    """
    Transforms correspondence clusters into a standard Krippendorff reliability matrix.

    Args:
        image_data (Dict[str, Any]): Preprocessed image data containing 'rater_list'.
        correspondence_clusters (List[Tuple[int, ...]]): Grouped annotation IDs.

    Returns:
        np.ndarray: Reliability matrix of shape (num_raters, num_clusters).
            Rows represent raters, columns represent units (clusters).
            Values are category IDs or -1.0 for missing detections.
    """
    rater_list = image_data['rater_list']
    num_raters = len(rater_list)
    num_clusters = len(correspondence_clusters)
    if num_clusters == 0:
        return np.array([[] for _ in range(num_raters)], dtype=float)

    # Note: -1.0 is used here as the "Existence Disagreement" value (rater missed the object)
    reliability_data = np.full((num_raters, num_clusters), -1.0, dtype=float)
    rater_to_idx = {rater_id: i for i, rater_id in enumerate(rater_list)}
    ann_id_to_annotation = {ann['id']: ann for rater_anns in image_data['annotations_by_rater'].values() for ann in rater_anns}

    for j, cluster in enumerate(correspondence_clusters):
        for ann_id in cluster:
            annotation = ann_id_to_annotation.get(ann_id)
            if annotation and annotation['rater_id'] in rater_to_idx:
                reliability_data[rater_to_idx[annotation['rater_id']], j] = annotation['category_id']
    return reliability_data
