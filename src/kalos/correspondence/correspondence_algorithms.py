"""
Core correspondence algorithms for geometric matching in Inter-Annotator Agreement.
Includes data preprocessing, pairwise score precomputation, and implementations 
for Greedy, Hungarian/SHM, AHC, and MGM matching strategies.
"""

import numpy as np
import argparse
import json
import logging
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from scipy.cluster.hierarchy import linkage, fcluster
from typing import Set, Optional

import itertools
from typing import Dict, List, Any, Tuple, Callable

from kalos.iaa.similarity_functions import SIMILARITY_FUNCTIONS
import pylibmgm

from tqdm import tqdm

logger = logging.getLogger(__name__)

# --- 1. Annotation Loader ---
def load_annotations(file_path: str) -> Dict[str, Any]:
    """
    Loads a COCO-style JSON annotation file.

    Args:
        file_path (str): The path to the JSON annotation file.

    Returns:
        Dict[str, Any]: The loaded annotation data as a Python dictionary.
    """
    logger.info(f"Loading annotations from: {file_path}")
    with open(file_path, 'r') as f:
        data = json.load(f)
    logger.debug("Annotations loaded successfully.")
    return data

# --- 2. Data Pre-processing ---
def _preprocess_coco(coco_data: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    """
    Pre-processes raw COCO data to group annotations by image and then by rater.
    Supports both standard List rater_list and Dict session-based rater_list.

    This function restructures the data to make it easier to access all annotations
    for a specific image, subdivided by the annotator who created them.

    Args:
        coco_data (Dict[str, Any]): The raw data loaded from the COCO-style JSON file.

    Returns:
        Dict[int, Dict[str, Any]]: A dictionary where each key is an `image_id`.
        The value is another dictionary containing the image's 'file_name',
        'rater_list', and a dictionary of 'annotations_by_rater'.
    """
    processed_data = {}

    # First, create a base structure for each image
    image_id_map = {img['id']: img for img in coco_data['images']}
    for image_id, img_info in image_id_map.items():
        if "rater_list" not in img_info:
            raise ValueError(f"Image {image_id} missing mandatory attribute 'rater_list'.")
        
        raw_list = img_info['rater_list']
        
        # Branch 1: Standard List format
        if isinstance(raw_list, list):
            flattened_list = raw_list
            image_session_mode = False
        # Branch 2: Dictionary session format
        elif isinstance(raw_list, dict):
            # Flatten dict { "Rater": [1, 2] } into ["Rater (S1)", "Rater (S2)"]
            flattened_list = []
            for rater_id, sessions in raw_list.items():
                for s_id in sessions:
                    flattened_list.append(f"{rater_id} (S{s_id})")
            image_session_mode = True
        else:
            raise TypeError(f"Invalid rater_list type for image {image_id}. Expected list or dict.")

        processed_data[image_id] = {
            'file_name': img_info['file_name'],
            'rater_list': flattened_list,
            'is_session_mode': image_session_mode,
            'annotations_by_rater': defaultdict(list)
        }

    # Now, populate the structure with annotations
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in processed_data:
            continue
            
        img_meta = image_id_map[image_id]
        width, height = img_meta['width'], img_meta['height']
        
        if "rater_id" not in ann:
            raise ValueError(f"Annotation {ann.get('id')} missing mandatory attribute 'rater_id'.")

        # Handle Identity Transformation
        rater_id = ann['rater_id']
        if processed_data[image_id]['is_session_mode']:
            s_id = ann.get('session_id')
            if s_id is None:
                raise ValueError(f"Session-mode detected for image {image_id}, but annotation {ann.get('id')} is missing 'session_id'.")
            # Map to the flattened identity
            internal_identity = f"{rater_id} (S{s_id})"
            # Update the annotation object to reflect its virtual session identity
            ann['rater_id'] = internal_identity
        else:
            internal_identity = rater_id

        # Relative coordinate conversion (bbox)
        if "bbox" in ann:
            bbox = ann['bbox']
            bbox[0] /= width
            bbox[1] /= height
            bbox[2] /= width
            bbox[3] /= height
        # Relative coordinate conversion (segmentation)
        if "segmentation" in ann:
            ann['segmentation'] = [
                [
                    coord / width if i % 2 == 0 else coord / height
                    for i, coord in enumerate(polygon)
                ]
                for polygon in ann['segmentation']
            ]
        if "keypoints" in ann:
            # normalize coco keypoints
            keypoints = ann['keypoints']
            # keypoints are xyv, where v is visibility.
            # v=0: not labeled (x=y=0), v=1: labeled but not visible, v=2: labeled and visible
            for i in range(0, len(keypoints), 3):
                if keypoints[i+2] > 0: # only normalize labeled keypoints
                    keypoints[i] /= width
                    keypoints[i+1] /= height

        # Only add if the rater/session is in the assigned list for this image
        if internal_identity in processed_data[image_id]['rater_list']:
            processed_data[image_id]['annotations_by_rater'][internal_identity].append(ann)
            
    return processed_data

def _preprocess_lidc_idri_data(lidc_idri_data: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    """
    Pre-processes raw LIDC-IDRI data into the internal standardized format.

    Args:
        lidc_idri_data (Dict[str, Any]): Raw LIDC-IDRI dictionary.

    Returns:
        Dict[int, Dict[str, Any]]: Preprocessed data structure compatible with KaLOS.
    """
    preprocess_data = {}

    for study_instance_uid, values in lidc_idri_data.items():
        # don't use case id, it is sometimes used multiple times
        preprocess_data[study_instance_uid] = {
            "file_name": values["file_paths"][0], # only first file extracted
            "rater_list": list(values["annotators"].keys()),
            "annotations_by_rater": defaultdict(list)
        }

    # second pass
    max_z = {}
    min_z = {}
    for study_instance_uid, values in lidc_idri_data.items():
        max_z[study_instance_uid] = float("-inf")
        min_z[study_instance_uid] = float("inf")
        for rater_id, annotations in values["annotators"].items():
            for annotation in annotations:
                for contour in annotation["contours"]:
                    max_z[study_instance_uid] = max(max_z[study_instance_uid], float(contour["z_position"]))
                    min_z[study_instance_uid] = min(min_z[study_instance_uid], float(contour["z_position"]))

    ann_id = 0
    # third pass populate the dictonary with annotations
    for study_instance_uid, values in lidc_idri_data.items():
        width, height, depth = values["width"], values["height"], values["depth"]
        z_range = max_z[study_instance_uid] - min_z[study_instance_uid]
        for rater_id, annotations in values["annotators"].items():
            # create empty dict
            preprocess_data[study_instance_uid]["annotations_by_rater"][rater_id] = []
            # fill with annotation data
            for annotation in annotations:
                for contour in annotation["contours"]:
                    if z_range > 0:
                        contour["z_position"] = (float(contour["z_position"]) - min_z[study_instance_uid]) / z_range
                    else:
                        contour["z_position"] = 0.0
                    contour["points"] = [[point[0] / width, point[1] / height] for point in contour["points"]]
                    assert 0 <= contour["z_position"] <= 1.0
                    assert all(0 <= x <= 1 for row in contour["points"] for x in row)
                ann = {"category_id": 1, "segmentation_3d": annotation["contours"], "id": ann_id, "rater_id": rater_id}
                ann_id += 1
                preprocess_data[study_instance_uid]["annotations_by_rater"][rater_id].append(
                    ann
                )

    return preprocess_data

def preprocess_data(annotation_data: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    """
    Pre-processes raw annotation data to group annotations by image and then by rater.

    This function inspects the data format and dispatches to the appropriate
    pre-processing function. Currently supports COCO and LIDC-IDRI formats.

    Args:
        annotation_data (Dict[str, Any]): The raw data loaded from the JSON file.

    Returns:
        Dict[int, Dict[str, Any]]: A dictionary where each key is an `image_id`.
        The value is another dictionary containing the image's 'file_name',
        'rater_list', and a dictionary of 'annotations_by_rater'.
    """
    logger.info("Preprocessing data...")

    # --- Data Format Dispatcher ---
    # Check for a key that is highly specific to the COCO format.
    if 'images' in annotation_data and 'annotations' in annotation_data:
        logger.debug("   - Detected COCO data format.")
        processed_data = _preprocess_coco(annotation_data)
    elif len(annotation_data) > 0 and "case_id" in next(iter(annotation_data.values())) and "study_instance_uid" in next(iter(annotation_data.values())):
        processed_data = _preprocess_lidc_idri_data(annotation_data)
    else:
        raise NotImplementedError("Unsupported data format. Only COCO-style JSON is currently supported.")

    logger.debug(f"Preprocessing complete. Found data for {len(processed_data)} images.")
    return processed_data

# --- 3. Precompute Pairwise Scores ---
def precompute_pairwise_scores(
        image_data: Dict[str, Any],
        threshold_func: Callable,
        similarity_threshold: float
    ) -> Dict[Tuple[int, int], Tuple[float, Dict, Dict]]:
    """
    Iterates through all unique annotation pairs once and precomputes similarity scores.
    This is done to prevent repeated cost computations.

    Args:
        image_data (Dict[str, Any]): Preprocessed data for a single image.
        threshold_func (Callable): Function to calculate similarity (e.g., IoU).
        similarity_threshold (float): Minimum similarity to keep the pair.

    Returns:
        Dict[Tuple[int, int], Tuple[float, Dict, Dict]]: A map from sorted (ann_id1, ann_id2) 
            to a tuple of (score, ann1, ann2).
    """
    annotations_by_rater = image_data['annotations_by_rater']
    raters = list(annotations_by_rater.keys())
    pairwise_scores = {}

    for rater1_id, rater2_id in itertools.combinations(raters, 2):
        for ann1 in annotations_by_rater[rater1_id]:
            for ann2 in annotations_by_rater[rater2_id]:
                score = threshold_func(ann1, ann2)
                if score >= similarity_threshold:
                    # Store with sorted IDs to ensure consistent key format
                    key = tuple(sorted((ann1['id'], ann2['id'])))
                    pairwise_scores[key] = (score, ann1, ann2) # Store score and anns
    return pairwise_scores

# --- 4. Cost and Threshold Functions ---

THRESHOLD_FUNCTIONS = {
    "bbox_iou_similarity": SIMILARITY_FUNCTIONS["bbox_iou_similarity"],
    "segm_iou_similarity": SIMILARITY_FUNCTIONS["segm_iou_similarity"],
    "segm_giou_similarity": SIMILARITY_FUNCTIONS["segm_giou_similarity"],
    "3D_IoU_similarity": SIMILARITY_FUNCTIONS["3D_IoU_similarity"],
    "in-mpjpe_similarity": SIMILARITY_FUNCTIONS["in-mpjpe_similarity"],
    "centroid_similarity": SIMILARITY_FUNCTIONS["centroid_similarity"],
}

COST_FUNCTIONS = {
    "negative_score": lambda score, ann1, ann2: -score,
    "category_lenient": lambda score, ann1, ann2: -score - 1.0 if ann1.get('category_id') == ann2.get('category_id') else -score,
}

# --- 5. Instance Correspondence Matching Functions ---
def match_greedy(
        image_data: Dict[str, Any],
        pairwise_scores: Dict[Tuple[int, int], Tuple[float, Dict, Dict]],
        cost_func: Callable,
        similarity_threshold: float
    ) -> List[Tuple[int, ...]]:
    """
    Performs instance correspondence matching using a greedy algorithm.

    Applies the cost function to all valid pairs, sorts them by cost ascending,
    and iteratively builds correspondence clusters while enforcing rater uniqueness.

    Args:
        image_data (Dict[str, Any]): The preprocessed data for a single image.
        pairwise_scores (Dict): A dictionary of pre-computed similarity scores.
        cost_func (Callable): A function that converts a score into a cost.
        similarity_threshold (float): Unused by this method, kept for a consistent interface.

    Returns:
        List[Tuple[int, ...]]: A list of tuples representing correspondence clusters.
    """
    # 0. Check if only allowed Id's are included in the matching
    allowed_ids = {ann['id'] for anns in image_data['annotations_by_rater'].values() for ann in anns}
    included_ids = {id_ for pair in pairwise_scores.keys() for id_ in pair}
    assert included_ids.issubset(allowed_ids), "Set contains invalid ID's."
    # 1. Apply the cost function to all valid pairs
    all_pairs_with_cost = [
        (cost_func(score, ann1, ann2), ann1, ann2)
        for score, ann1, ann2 in pairwise_scores.values()
    ]

    # 2. Greedy sorting: sort all possible matches by cost, ascending (lower is better)
    all_pairs_with_cost.sort(key=lambda x: x[0])

    # 3. Iteratively build clusters with constraints
    # These structures track the clusters and enforce the matching rules
    ann_to_cluster_id: Dict[int, int] = {}
    clusters: List[set] = []
    # Create a map of rater IDs for the potential new cluster
    cluster_raters: List[set] = []

    for iou, ann1, ann2 in all_pairs_with_cost:
        ann1_id, rater1_id = ann1['id'], ann1['rater_id']
        ann2_id, rater2_id = ann2['id'], ann2['rater_id']

        # --- Clustering Logic ---
        c1_id = ann_to_cluster_id.get(ann1_id)
        c2_id = ann_to_cluster_id.get(ann2_id)

        # --- UNIFIED CONSTRAINT CHECK ---
        # If they are already in the same cluster, there's nothing to do.
        if c1_id is not None and c1_id == c2_id:
            continue

        r_ids_in_c1 = cluster_raters[c1_id] if c1_id is not None else set()
        r_ids_in_c2 = cluster_raters[c2_id] if c2_id is not None else set()

        # Case 1: Merging two existing, different clusters. They cannot share any raters.
        if c1_id is not None and c2_id is not None and c1_id != c2_id:
            if r_ids_in_c1.intersection(r_ids_in_c2):
                continue  # This merge is invalid.
        # Case 2: Adding ann2 to ann1's cluster. The rater of ann2 cannot already be in the cluster.
        elif c1_id is not None and c2_id is None:
            if rater2_id in r_ids_in_c1:
                continue  # This addition is invalid.
        # Case 3: Adding ann1 to ann2's cluster. The rater of ann1 cannot already be in the cluster.
        elif c1_id is None and c2_id is not None:
            if rater1_id in r_ids_in_c2:
                continue  # This addition is invalid.
        # Case 4 (Implicit): Creating a new cluster. This is always valid as rater1_id != rater2_id.

        if c1_id is None and c2_id is None:
            # Neither annotation is in a cluster yet. Create a new one.
            new_cluster = {ann1_id, ann2_id}
            new_cluster_id = len(clusters)
            clusters.append(new_cluster)
            cluster_raters.append({rater1_id, rater2_id})
            ann_to_cluster_id[ann1_id] = new_cluster_id
            ann_to_cluster_id[ann2_id] = new_cluster_id
        elif c1_id is not None and c2_id is None:
            # ann1 is in a cluster, ann2 is not. Add ann2 to ann1's cluster.
            clusters[c1_id].add(ann2_id)
            cluster_raters[c1_id].add(rater2_id)
            ann_to_cluster_id[ann2_id] = c1_id
        elif c1_id is None and c2_id is not None:
            # ann2 is in a cluster, ann1 is not. Add ann1 to ann2's cluster.
            clusters[c2_id].add(ann1_id)
            cluster_raters[c2_id].add(rater1_id)
            ann_to_cluster_id[ann1_id] = c2_id
        elif c1_id != c2_id:
            # Both are in different clusters. Merge the smaller into the larger.
            cluster1_content = clusters[c1_id]
            cluster2_content = clusters[c2_id]
            raters1_content = cluster_raters[c1_id]
            raters2_content = cluster_raters[c2_id]

            # Merge smaller cluster into the larger one for efficiency
            if len(cluster1_content) < len(cluster2_content):
                c1_id, c2_id = c2_id, c1_id  # Swap them
                cluster1_content, cluster2_content = cluster2_content, cluster1_content
                raters1_content, raters2_content = raters2_content, raters1_content

            cluster1_content.update(cluster2_content)
            raters1_content.update(raters2_content)
            # Update all annotations from the merged cluster
            for ann_id in cluster2_content:
                ann_to_cluster_id[ann_id] = c1_id
            # Mark old cluster as empty to be filtered out later
            clusters[c2_id] = set()
            cluster_raters[c2_id] = set()
        # If c1_id == c2_id, they are already in the same cluster, do nothing.

    # 4. Finalize the output format
    # Filter out any empty sets that resulted from merging
    final_clusters = [c for c in clusters if c]

    correspondence_clusters = [tuple(sorted(list(cluster))) for cluster in final_clusters]

    all_ann_ids = {ann['id'] for rater_anns in image_data['annotations_by_rater'].values() for ann in rater_anns}
    matched_ann_ids = set(ann_to_cluster_id.keys())
    singleton_ids = all_ann_ids - matched_ann_ids
    correspondence_clusters.extend([(sid,) for sid in singleton_ids])

    # --- Integrity Checks ---
    # 1. Check for completeness
    input_ids = {ann['id'] for rater_anns in image_data['annotations_by_rater'].values() for ann in rater_anns}
    output_ids = {ann_id for cluster in correspondence_clusters for ann_id in cluster}
    assert input_ids == output_ids, f"[Greedy] Completeness Check Failed. Missing: {input_ids - output_ids}, Extra: {output_ids - input_ids}"

    # 2. Check for valid cluster size
    max_cluster_size = len(image_data["rater_list"])
    assert all(len(c) <= max_cluster_size for c in correspondence_clusters), f"[Greedy] Size Check Failed. Found cluster too large for {max_cluster_size} raters."

    # 3. Check for unique raters within each cluster
    ann_id_to_rater_id = {ann['id']: ann['rater_id'] for anns in image_data['annotations_by_rater'].values() for ann in anns}
    for cluster in correspondence_clusters:
        raters_in_cluster = [ann_id_to_rater_id[ann_id] for ann_id in cluster]
        assert len(raters_in_cluster) == len(set(raters_in_cluster)), f"[Greedy] Uniqueness Check Failed. Cluster {cluster} has duplicate raters."


    return correspondence_clusters

def match_shm(
        image_data: Dict[str, Any],
        pairwise_scores: Dict[Tuple[int, int], Tuple[float, Dict, Dict]],
        cost_func: Callable,
        similarity_threshold: float
    ) -> List[Tuple[int, ...]]:
    """
    Performs instance correspondence matching using Sequential Hungarian Matching (SHM).

    Builds a correspondence matrix by adding one rater at a time and matching
    them against established clusters using the Sequential Hungarian algorithm (Tschirschwitz et al. 2022).

    Args:
        image_data (Dict[str, Any]): The preprocessed data for a single image.
        pairwise_scores (Dict): Pre-computed similarity scores for annotation pairs.
        cost_func (Callable): A function that converts a score into a cost.
        similarity_threshold (float): Unused by this method, kept for a consistent interface.

    Returns:
        List[Tuple[int, ...]]: A list of tuples representing correspondence clusters.
    """
    # 0. Check if only allowed Id's are included in the matching
    allowed_ids = {ann['id'] for anns in image_data['annotations_by_rater'].values() for ann in anns}
    included_ids = {id_ for pair in pairwise_scores.keys() for id_ in pair}
    assert included_ids.issubset(allowed_ids), "Set contains invalid ID's."

    all_annotations = image_data['annotations_by_rater']
    raters = list(all_annotations.keys())
    # -----------------------------------
    # WARNING: This is different to the original implementation.
    # WARNING: It is there to ensure that permutation stability can be tested.
    # WARNING: In the original implementations the annotators are sorted lexicographically
    # -----------------------------------

    # nothing to do
    if not all_annotations:
        return []
    if len(raters) < 2:
        return [(ann['id'],) for ann in all_annotations[raters[0]]]

    # --- Helper function for the core bipartite matching step ---
    def _run_bipartite_hungarian(
        group1: List[Dict],
        group2: List[Dict],
        pairwise_scores: Dict[Tuple[int, int], Tuple[float, Dict, Dict]],
        cost_func: Callable
    ) -> List[Tuple[Dict, Dict]]:
        if not group1 or not group2:
            return []

        # 1. Mutual Filtering: Keep only annotations that have at least one valid partner.
        g1_ids = {ann['id'] for ann in group1}
        g2_ids = {ann['id'] for ann in group2}

        valid_g1_ids = set()
        valid_g2_ids = set()

        for (id1, id2), _ in pairwise_scores.items():
            # Check for both possible orderings since the key is sorted
            if id1 in g1_ids and id2 in g2_ids:
                valid_g1_ids.add(id1)
                valid_g2_ids.add(id2)
            elif id2 in g1_ids and id1 in g2_ids:
                valid_g1_ids.add(id2)
                valid_g2_ids.add(id1)

        filtered_g1 = [ann for ann in group1 if ann['id'] in valid_g1_ids]
        filtered_g2 = [ann for ann in group2 if ann['id'] in valid_g2_ids]

        # 2. Padding: Add dummy 'None' entries to the shorter list to make the problem square.
        len1, len2 = len(filtered_g1), len(filtered_g2)
        if len1 < len2:
            filtered_g1.extend([None] * (len2 - len1))
        elif len2 < len1:
            filtered_g2.extend([None] * (len1 - len2))

        if not filtered_g1:
            return []

        # 3. Build Square Cost Matrix: Use a high cost for dummy assignments.
        # A neutral high cost is 0, since our cost functions produce negative values for good matches.
        PENALTY_COST = 100.0
        size = len(filtered_g1)
        cost_matrix = np.full((size, size), PENALTY_COST)

        for i, ann1 in enumerate(filtered_g1):
            # Skip dummy rows created by padding
            if ann1 is None:
                continue
            for j, ann2 in enumerate(filtered_g2):
                # Skip dummy columns
                if ann2 is None:
                    continue

                key = tuple(sorted((ann1['id'], ann2['id'])))
                if key in pairwise_scores:
                    score, _, _ = pairwise_scores[key]
                    cost_matrix[i, j] = cost_func(score, ann1, ann2)

        # 4. Solve the now-guaranteed-solvable square assignment problem.
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # 5. Extract valid pairs, ensuring they are not dummy assignments and have a finite cost.
        matched_pairs = []
        for i in range(len(row_ind)):
            cost = cost_matrix[row_ind[i], col_ind[i]]

            # Valid matches have costs lower than the PENALTY COST
            if cost < PENALTY_COST:
                ann1 = filtered_g1[row_ind[i]]
                ann2 = filtered_g2[col_ind[i]]
                # Final check to ensure we didn't match two real annotations with a fallback cost
                # and that we aren't matching a dummy 'None' entry.
                if ann1 is not None and ann2 is not None:
                    matched_pairs.append((ann1, ann2))

        return matched_pairs

    # --- Recursive function to build the correspondence matrix ---
    def _build_matrix_recursively(
            coincidence_matrix: List[List[Optional[Dict]]],
            matched_raters: Set[str],
    ) -> List[List[Optional[Dict]]]:
        # Base case: all raters have been processed
        if len(matched_raters) == len(raters):
            return coincidence_matrix

        # Identify the next rater to add based on the original order
        next_rater_idx = len(matched_raters)
        next_rater_id = raters[next_rater_idx]

        # Initialize the new rater's row with None placeholders
        num_cols = len(coincidence_matrix[0]) if coincidence_matrix and coincidence_matrix[0] else 0
        coincidence_matrix.append([None] * num_cols)
        unmatched_ann_list = list(all_annotations[next_rater_id])

        # Attempt to match the new rater against all previously matched raters
        for prev_rater_id in matched_raters:
            prev_rater_idx = raters.index(prev_rater_id)

            # Find available annotations from the previous rater to match against
            available_slots = []
            for col_idx, ann in enumerate(coincidence_matrix[prev_rater_idx]):
                if ann is not None and coincidence_matrix[next_rater_idx][col_idx] is None:
                    available_slots.append(ann)

            if not available_slots or not unmatched_ann_list:
                continue

            # Run bipartite matching
            newly_matched_pairs = _run_bipartite_hungarian(
                available_slots, unmatched_ann_list, pairwise_scores, cost_func
            )

            if not newly_matched_pairs:
                continue

            # Update the matrix with the new matches
            # Use a set of IDs for efficient lookup and filtering.
            matched_ids_from_current_rater = set()
            for prev_ann, new_ann in newly_matched_pairs:
                col_to_update = coincidence_matrix[prev_rater_idx].index(prev_ann)
                coincidence_matrix[next_rater_idx][col_to_update] = new_ann
                matched_ids_from_current_rater.add(new_ann['id'])

            # Update the list of remaining unmatched annotations using their IDs.
            if matched_ids_from_current_rater:
                unmatched_ann_list = [
                    ann for ann in unmatched_ann_list if ann['id'] not in matched_ids_from_current_rater
                ]

        # Handle "leftover" annotations by adding new columns
        if unmatched_ann_list:
            leftovers = sorted(unmatched_ann_list, key=lambda x: x['id'])
            for leftover_ann in leftovers:
                for row_idx in range(len(coincidence_matrix)):
                    if row_idx == next_rater_idx:
                        coincidence_matrix[row_idx].append(leftover_ann)
                    else:
                        coincidence_matrix[row_idx].append(None)

        # Recurse for the next rater
        matched_raters.add(next_rater_id)
        return _build_matrix_recursively(coincidence_matrix, matched_raters)

    # --- Main execution flow ---
    # 1. Initialize the matrix with the first rater's annotations
    initial_rater_id = raters[0]
    initial_annotations = sorted(all_annotations[initial_rater_id], key=lambda x: x['id'])
    matrix = [initial_annotations]

    # 2. Start the recursive building process
    final_matrix = _build_matrix_recursively(matrix, {initial_rater_id})

    # 3. Convert the final matrix into the desired output format
    correspondence_clusters = []
    assert final_matrix or final_matrix[0]

    num_clusters = len(final_matrix[0])
    for col_idx in range(num_clusters):
        cluster = []
        for row_idx in range(len(final_matrix)):
            ann = final_matrix[row_idx][col_idx]
            if ann is not None:
                cluster.append(ann['id'])
        if cluster:
            correspondence_clusters.append(tuple(sorted(cluster)))

    # --- Integrity Checks ---
    # 1. Check for completeness
    input_ids = {ann['id'] for rater_anns in image_data['annotations_by_rater'].values() for ann in rater_anns}
    output_ids = {ann_id for cluster in correspondence_clusters for ann_id in cluster}
    assert input_ids == output_ids, f"[SHM] Completeness Check Failed. The core logic dropped annotations. Missing: {input_ids - output_ids}"

    # 2. Check for valid cluster size
    max_cluster_size = len(image_data["rater_list"])
    assert all(len(c) <= max_cluster_size for c in correspondence_clusters), f"[SHM] Size Check Failed. Found cluster too large for {max_cluster_size} raters."

    # 3. Check for unique raters within each cluster
    ann_id_to_rater_id = {ann['id']: ann['rater_id'] for anns in image_data['annotations_by_rater'].values() for ann in anns}
    for cluster in correspondence_clusters:
        raters_in_cluster = [ann_id_to_rater_id[ann_id] for ann_id in cluster]
        assert len(raters_in_cluster) == len(set(raters_in_cluster)), f"[SHM] Uniqueness Check Failed. Cluster {cluster} has duplicate raters."

    return correspondence_clusters

def match_ahc(
        image_data: Dict[str, Any],
        pairwise_scores: Dict[Tuple[int, int], Tuple[float, Dict, Dict]],
        cost_func: Callable,
        similarity_threshold: float,
    ) -> List[Tuple[int, ...]]:
    """
    Performs instance correspondence matching using Agglomerative Hierarchical Clustering (AHC).

    Clusters annotations based on geometric distance, enforcing rater uniqueness 
    constraints as described in Amgad et al. (2022).

    Args:
        image_data (Dict[str, Any]): The preprocessed data for a single image.
        pairwise_scores (Dict): Pre-computed similarity scores for annotation pairs.
        cost_func (Callable): A function that converts a score into a cost.
        similarity_threshold (float): The similarity score threshold used for the clustering cut-off.

    Returns:
        List[Tuple[int, ...]]: A list of tuples representing correspondence clusters.
    """
    # 0. Check if only allowed Id's are included in the matching
    allowed_ids = {ann['id'] for anns in image_data['annotations_by_rater'].values() for ann in anns}
    included_ids = {id_ for pair in pairwise_scores.keys() for id_ in pair}
    assert included_ids.issubset(allowed_ids), "Set contains invalid ID's."

    all_annotations = [
        ann for rater_anns in image_data['annotations_by_rater'].values() for ann in rater_anns
    ]

    # nothing to do
    if not all_annotations:
        return []

    num_annotations = len(all_annotations)
    PENALTY_DISTANCE = 100.0  # A large penalty for pairs that should not be clustered

    # 1. Calculate all potential costs first
    all_costs = []
    # This list will hold costs in the same order as the condensed matrix
    for i in range(num_annotations):
        for j in range(i + 1, num_annotations):
            ann1, ann2 = all_annotations[i], all_annotations[j]
            cost = PENALTY_DISTANCE  # Default to penalty

            if ann1['rater_id'] != ann2['rater_id']:
                key = tuple(sorted((ann1['id'], ann2['id'])))
                # in case the combination is not found the default penalty was already set.
                if key in pairwise_scores:
                    score, _, _ = pairwise_scores[key]
                    cost = cost_func(score, ann1, ann2)

            all_costs.append(cost)

    # no elements in range
    if not all_costs:
        return [(ann['id'],) for ann in all_annotations]

    # 2. Calculate Cost offset
    min_cost = min(all_costs)
    cost_offset = 0.0
    if min_cost < 0:
        cost_offset = -min_cost # e.g., if min_cost is -2, offset is 2

    condensed_dist_matrix = [cost + cost_offset for cost in all_costs]

    # 3. Perform Agglomerative Hierarchical Clustering
    # The 'complete' method calculates the maximum distance between all
    # observations of the two sets of observations.
    linkage_matrix = linkage(condensed_dist_matrix, method='complete')

    # 4. Form final flat clusters by dynamically calculating the cut-off distance.
    # We apply the cost function to the threshold value to find the equivalent distance + the cost offset.
    distance_threshold = cost_func(similarity_threshold, {}, {}) + cost_offset
    cluster_labels = fcluster(linkage_matrix, t=distance_threshold, criterion='distance')

    # Group annotation IDs by their assigned cluster label
    clusters = defaultdict(list)
    for i, label in enumerate(cluster_labels):
        clusters[label].append(all_annotations[i]['id'])

    # Format the output, initially filtering out single-element clusters
    matched_clusters = [
        tuple(sorted(ids)) for ids in clusters.values() if len(ids) > 1
    ]

    all_ann_ids = {ann['id'] for ann in all_annotations}
    matched_ann_ids = {item for cluster in matched_clusters for item in cluster}
    singleton_ids = all_ann_ids - matched_ann_ids
    matched_clusters.extend([(sid,) for sid in singleton_ids])

    # --- Integrity Checks ---
    # 1. Check for completeness
    output_ids = {ann_id for cluster in matched_clusters for ann_id in cluster}
    assert all_ann_ids == output_ids, f"[AHC] Completeness Check Failed. Missing: {all_ann_ids - output_ids}, Extra: {output_ids - all_ann_ids}"

    # 2. Check for valid cluster size
    max_cluster_size = len(image_data["rater_list"])
    assert all(len(c) <= max_cluster_size for c in matched_clusters), f"[AHC] Size Check Failed. Found cluster too large for {max_cluster_size} raters."

    # 3. Check for unique raters within each cluster
    ann_id_to_rater_id = {ann['id']: ann['rater_id'] for anns in image_data['annotations_by_rater'].values() for ann in anns}
    for cluster in matched_clusters:
        raters_in_cluster = [ann_id_to_rater_id[ann_id] for ann_id in cluster]
        assert len(raters_in_cluster) == len(set(raters_in_cluster)), f"[AHC] Uniqueness Check Failed. Cluster {cluster} has duplicate raters."

    return matched_clusters

def match_mgm(
        image_data: Dict[str, Any],
        pairwise_scores: Dict[Tuple[int, int], Tuple[float, Dict, Dict]],
        cost_func: Callable,
        similarity_threshold: float,
    ) -> List[Tuple[int, ...]]:
    """
    Performs instance correspondence matching using Multi-Graph Matching (MGM).

    Uses the pylibmgm library (Python binding for Kahl et al., 2025) to solve 
    global correspondence across all raters simultaneously.

    Args:
        image_data (Dict[str, Any]): The preprocessed data for a single image.
        pairwise_scores (Dict): Pre-computed similarity scores for annotation pairs.
        cost_func (Callable): A function that converts a score into a cost.
        similarity_threshold (float): Unused, kept for a consistent interface.

    Returns:
        List[Tuple[int, ...]]: A list of tuples containing annotation IDs of matched sets.
    """
    # 0. Check if only allowed Id's are included in the matching
    allowed_ids = {ann['id'] for anns in image_data['annotations_by_rater'].values() for ann in anns}
    included_ids = {id_ for pair in pairwise_scores.keys() for id_ in pair}
    assert included_ids.issubset(allowed_ids), "Set contains invalid ID's."

    all_annotations_by_rater = image_data['annotations_by_rater']
    raters = list(all_annotations_by_rater.keys())
    num_raters = len(raters)

    # nothing to do
    if not raters:
        return []

    if num_raters < 2:
         return [(ann['id'],) for ann in all_annotations_by_rater[raters[0]]]

    # --- 1. Create Graphs and Mappings ---
    graphs = {}
    # Nested dict mapping: {rater_id: {annotation_id: node_id}}
    ann_to_node_map = defaultdict(dict)
    # Nested dict mapping: {rater_id: {node_id: annotation_id}}
    node_to_ann_map = defaultdict(dict)

    for i, rater_id in enumerate(raters):
        annotations = all_annotations_by_rater[rater_id]
        graphs[rater_id] = pylibmgm.Graph(i, len(annotations))
        for node_id, ann in enumerate(annotations):
            ann_id = ann['id']
            ann_to_node_map[rater_id][ann_id] = node_id
            node_to_ann_map[rater_id][node_id] = ann_id

    # --- 2. Build the main MgmModel from pairwise GmModels ---
    mgm_model = pylibmgm.MgmModel()

    # Iterate through all unique pairs of graphs (raters).
    for i in range(num_raters):
        for j in range(i + 1, num_raters):
            rater1_id = raters[i]
            rater2_id = raters[j]

            graph1 = graphs[rater1_id]
            graph2 = graphs[rater2_id]

            assignments = []
            for ann1 in all_annotations_by_rater[rater1_id]:
                for ann2 in all_annotations_by_rater[rater2_id]:
                    key = tuple(sorted((ann1['id'], ann2['id'])))
                    if key in pairwise_scores:
                        score, _, _ = pairwise_scores[key]
                        cost = cost_func(score, ann1, ann2)
                        node1_id = ann_to_node_map[rater1_id][ann1['id']]
                        node2_id = ann_to_node_map[rater2_id][ann2['id']]
                        assignments.append((node1_id, node2_id, cost))

            gm_model = pylibmgm.GmModel(graph1, graph2, len(assignments) or 1, 0)
            for node1_id, node2_id, cost in assignments:
                gm_model.add_assignment(node1_id, node2_id, cost)

            # Add the fully constructed pairwise model to the main MGM model
            mgm_model.add_model(gm_model)

    solution = pylibmgm.solver.solve_mgm(mgm_model)

    # --- 4. Parse and Format the Solution using .labeling() ---
    labeling = solution.labeling()
    if not labeling:
        all_ids = {ann['id'] for rater_anns in all_annotations_by_rater.values() for ann in rater_anns}
        return [(sid,) for sid in all_ids]

    ann_to_cluster_id = {}
    clusters = defaultdict(set)
    next_cluster_id = 0

    # Iterate through the pairwise matchings
    for (g1_idx, g2_idx), matches in labeling.items():
        rater1_id = raters[g1_idx]
        rater2_id = raters[g2_idx]

        for n1_idx, n2_idx in enumerate(matches):
            if n2_idx == -1:  # Skip unmatched nodes
                continue

            ann1_id = node_to_ann_map[rater1_id][n1_idx]
            ann2_id = node_to_ann_map[rater2_id][n2_idx]

            c1_id = ann_to_cluster_id.get(ann1_id)
            c2_id = ann_to_cluster_id.get(ann2_id)

            if c1_id is None and c2_id is None:
                # Neither annotation is in a cluster yet. Create a new one.
                clusters[next_cluster_id].update([ann1_id, ann2_id])
                ann_to_cluster_id[ann1_id] = next_cluster_id
                ann_to_cluster_id[ann2_id] = next_cluster_id
                next_cluster_id += 1
            elif c1_id is not None and c2_id is None:
                # ann1 is in a cluster, ann2 is new. Add ann2 to the cluster.
                clusters[c1_id].add(ann2_id)
                ann_to_cluster_id[ann2_id] = c1_id
            elif c1_id is None and c2_id is not None:
                # ann2 is in a cluster, ann1 is new. Add ann1 to the cluster.
                clusters[c2_id].add(ann1_id)
                ann_to_cluster_id[ann1_id] = c2_id
            elif c1_id != c2_id:
                # Both are in different clusters. Merge them.
                # Merge the smaller cluster into the larger one for efficiency.
                if len(clusters[c1_id]) < len(clusters[c2_id]):
                    c1_id, c2_id = c2_id, c1_id  # Swap them

                # Move all annotations from cluster 2 to cluster 1
                for ann_id in clusters[c2_id]:
                    ann_to_cluster_id[ann_id] = c1_id
                clusters[c1_id].update(clusters[c2_id])
                del clusters[c2_id]
            # If c1_id == c2_id, they are already linked. Do nothing.

    correspondence_clusters = [tuple(sorted(list(c))) for c in clusters.values()]

    all_ids = {ann['id'] for rater_anns in all_annotations_by_rater.values() for ann in rater_anns}
    matched_ids = {item for cluster in correspondence_clusters for item in cluster}
    singleton_ids = all_ids - matched_ids
    correspondence_clusters.extend([(sid,) for sid in singleton_ids])

    # --- Integrity Checks ---
    # 1. Check for completeness
    input_ids = {ann['id'] for rater_anns in image_data['annotations_by_rater'].values() for ann in rater_anns}
    output_ids = {ann_id for cluster in correspondence_clusters for ann_id in cluster}
    assert input_ids == output_ids, f"[MGM] Completeness Check Failed. Missing: {input_ids - output_ids}, Extra: {output_ids - input_ids}"

    # 2. Check for valid cluster size
    max_cluster_size = len(image_data["rater_list"])
    assert all(len(c) <= max_cluster_size for c in
               correspondence_clusters), f"[MGM] Size Check Failed. Found cluster too large for {max_cluster_size} raters."

    # 3. Check for unique raters within each cluster
    ann_id_to_rater_id = {ann['id']: ann['rater_id'] for anns in image_data['annotations_by_rater'].values() for ann in anns}
    for cluster in correspondence_clusters:
        raters_in_cluster = [ann_id_to_rater_id[ann_id] for ann_id in cluster]
        assert len(raters_in_cluster) == len(set(raters_in_cluster)), f"[MGM] Uniqueness Check Failed. Cluster {cluster} has duplicate raters."

    return correspondence_clusters

MATCHING_FUNCTIONS = {
    'greedy': match_greedy,
    'shm': match_shm,
    'ahc': match_ahc,
    'mgm': match_mgm,
}

# --- 6. Main Orchestration Function ---

def main(args: argparse.Namespace):
    """
    The main function to run the entire modular pipeline.
    This can be used for stand-alone usage

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    # --- 1. Dynamically select functions using the registry pattern ---
    try:
        threshold_function = THRESHOLD_FUNCTIONS[args.threshold_func]
        cost_function = COST_FUNCTIONS[args.cost_func]
        matching_function = MATCHING_FUNCTIONS[args.method]
    except KeyError as e:
        logger.error(f"Invalid function name provided: {e}. Please check your arguments.")
        return

    # --- 2. Load and preprocess data ---
    annotation_data = load_annotations(args.annotation_file)
    processed_data = preprocess_data(annotation_data)

    logger.info(f"  Running method: '{args.method}'")
    logger.info(f"   - Similarity Function: '{args.threshold_func}'")
    logger.info(f"   - Cost Function: '{args.cost_func}'")
    logger.info(f"   - Similarity Threshold: {args.similarity_threshold:.2f}")
    logger.info(f"       -> Valid Match Range: [{args.similarity_threshold:.2f}, 1.00] (Score >= {args.similarity_threshold:.2f} is accepted)")
    logger.info(f"       -> Interpretation: Higher score = Higher similarity (1.0 is perfect).")
    logger.info(f"   - Optimization Goal: Minimize Cost (Lower cost is better).")
    logger.info(f"       -> Logic: The solver minimizes cost to maximize similarity.")
    if args.cost_func == "negative_score":
        logger.info(f"       -> Mapping Examples:")
        logger.info(f"          * Perfect Match (Score 1.0)  -> Cost -1.0 (Lowest/Best)")
        logger.info(f"          * Minimal Match (Score {args.similarity_threshold:.2f}) -> Cost -{args.similarity_threshold:.2f} (Highest/Worst)")
    elif args.cost_func == "category_lenient":
        logger.info(f"       -> Mapping Examples:")
        logger.info(f"          * Perfect Match (Score 1.0, Same Class) -> Cost -2.0 (Lowest/Best)")
        logger.info(f"          * Perfect Match (Score 1.0, Diff Class) -> Cost -1.0")

    all_correspondences = {}
    for image_id, image_data in tqdm(processed_data.items(), desc="Processing Images"):
        # --- 3. Pre-compute scores once per image ---
        # This is the efficient pre-computation step you've already adopted.
        pairwise_scores = precompute_pairwise_scores(image_data, threshold_function, args.similarity_threshold)

        # --- 4. Run the selected matching function ---
        image_correspondences = matching_function(
            image_data=image_data,
            pairwise_scores=pairwise_scores,
            cost_func=cost_function,
            similarity_threshold=args.similarity_threshold
        )

        image_filename = image_data['file_name']
        all_correspondences[image_filename] = image_correspondences

    # --- 5. Display results ---
    logger.info("✅ Correspondence analysis complete.")
    if all_correspondences:
        sample_image = next(iter(all_correspondences.keys()))
        logger.info(f"Sample correspondence for '{sample_image}':")
        # To avoid a huge printout, let's show stats for the sample
        num_clusters = len(all_correspondences[sample_image])
        num_singletons = sum(1 for c in all_correspondences[sample_image] if len(c) == 1)
        logger.info(f"  - Found {num_clusters} total clusters.")
        logger.info(f"  - Found {num_singletons} singleton (unmatched) annotations.")

# --- 5. Argument Parsing and Execution ---
def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments for the modular script.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Run modular instance correspondence matching.")

    parser.add_argument("-a", "--annotation_file", type=str, required=True,
                        help="Path to the COCO-style JSON annotation file.")

    # Arguments for selecting the functions
    parser.add_argument("-m", "--method", type=str, required=True, choices=list(MATCHING_FUNCTIONS.keys()),
                        help="The correspondence matching method to use.")
    parser.add_argument("-tf", "--threshold_func", type=str, required=True, choices=list(THRESHOLD_FUNCTIONS.keys()),
                        help="The function to calculate similarity score (e.g., iou).")
    parser.add_argument("-cf", "--cost_func", type=str, required=True, choices=list(COST_FUNCTIONS.keys()),
                        help="The function to convert a score to a cost.")

    # Argument for the threshold value
    parser.add_argument("-tv", "--similarity_threshold", type=float, required=True,
                        help="The minimum similarity score to consider a valid match.")

    return parser.parse_args()


if __name__ == '__main__':
    arguments = parse_arguments()
    main(arguments)
