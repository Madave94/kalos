"""
Pipeline for calculating observed ($D_o$) and expected ($D_e$) disagreement distributions.
Used as a preliminary step for deriving principled localization thresholds.
"""

import json
import random
import logging
from collections import defaultdict
from typing import Dict, List, Any, Callable
from tqdm import tqdm

from kalos.correspondence.correspondence_algorithms import (
    preprocess_data,
    load_annotations,
    precompute_pairwise_scores,
)
from kalos.iaa.similarity_functions import SIMILARITY_FUNCTIONS
from kalos.utils.logging import setup_kalos_logging
from kalos.config import EmpiricalDisagreementConfig

logger = logging.getLogger(__name__)


# --- 1. D_o and D_e Calculation ---


def calculate_do_de(
    processed_data: Dict[int, Dict[str, Any]],
    similarity_function: Callable[[Dict, Dict], float]
) -> Dict[str, List[float]]:
    """
    Calculates observed (D_o) and expected (D_e) disagreement values as a
    minimization problem (distance = 1 - similarity).

    Args:
        processed_data (Dict[int, Dict[str, Any]]): Preprocessed annotation data 
            grouped by image and rater.
        similarity_function (Callable[[Dict, Dict], float]): The function used to 
            calculate the similarity between two geometric annotations.

    Returns:
        Dict[str, List[float]]: A dictionary containing two lists: 'd_o' (observed 
            disagreements) and 'd_e' (expected disagreements).

    Raises:
        ValueError: If fewer than 2 images are provided, making D_e calculation impossible.
    """
    d_o_values = []
    d_e_values = []
    all_image_ids = list(processed_data.keys())
    max_distance = 1.0 # All distances are normalized to [0, 1]
    similarity_to_distance_func = lambda x, y: max_distance - similarity_function(x, y)

    if len(all_image_ids) < 2:
        raise ValueError("Cannot calculate D_e, as it requires at least 2 images.")

    # --- D_o Calculation ---
    for image_id, image_data in tqdm(processed_data.items(), desc="Calculating D_o (Observed)"):
        raters = image_data['rater_list']
        if len(raters) < 2:
            continue

        # 1. Precompute all pairwise scores (similarities).
        pairwise_scores = precompute_pairwise_scores(image_data, similarity_to_distance_func, 0.0)

        # 2. For each annotation, find its minimum distance to each *other* rater's set.
        best_match_for_ann = defaultdict(dict)
        for rater in image_data["rater_list"]:
            for ann in image_data["annotations_by_rater"][rater]:
                for other_rater in image_data["rater_list"]:
                    if other_rater == rater:
                        continue
                    best_match_for_ann[ann["id"]][other_rater] = max_distance

        for distance, ann1, ann2 in pairwise_scores.values():
            if distance > max_distance:
                logger.warning(f"Calculated distance ({distance:.4f}) exceeded max_distance ({max_distance}). Clamping.")
                distance = max_distance
            id1, rater1 = ann1['id'], ann1['rater_id']
            id2, rater2 = ann2['id'], ann2['rater_id']

            best_match_for_ann[id1][rater2] = min(best_match_for_ann[id1][rater2], distance)
            best_match_for_ann[id2][rater1] = min(best_match_for_ann[id2][rater1], distance)

        d_o_values.extend(v for inner in best_match_for_ann.values() for v in inner.values())

    # --- D_e Calculation ---
    MAX_RETRIES = 10
    for ref_image_id, ref_image_data in tqdm(processed_data.items(), desc="Calculating D_e (Expected)"):
        other_image_ids = [img_id for img_id in all_image_ids if img_id != ref_image_id]
        ref_annotations_by_rater = ref_image_data['annotations_by_rater']

        for ref_rater_id, ref_annotations in ref_annotations_by_rater.items():
            if not ref_annotations:
                continue
            random_image_id = None
            for _ in range(MAX_RETRIES):
                candidate_image_id = random.choice(other_image_ids)
                if candidate_image_id != ref_image_id:
                    random_image_id = candidate_image_id
                    break

            if random_image_id is None:
                raise Exception(f"Could not select a different image for ref_image_id {ref_image_id} after {MAX_RETRIES} retries.")

            random_image_data = processed_data[random_image_id]
            random_raters = random_image_data['rater_list']

            random_rater_id = random.choice(random_raters)
            random_annotations = random_image_data['annotations_by_rater'][random_rater_id]
            if not random_annotations:
                d_e_values.extend(max_distance for _ in ref_annotations)
                continue

            temp_image_data = {
                'annotations_by_rater': {
                    'ref_rater': ref_annotations,
                    'rand_rater': random_annotations
                },
                'rater_list': ['ref_rater', 'rand_rater']
            }

            pairwise_scores = precompute_pairwise_scores(temp_image_data, similarity_to_distance_func, 0.0)

            min_scores = dict()
            for ref_ann in ref_annotations:
                min_scores[ref_ann["id"]] = max_distance
            for distance, ref_ann, rand_ann in pairwise_scores.values():
                if distance > max_distance:
                    logger.warning(f"Calculated distance ({distance:.4f}) exceeded max_distance ({max_distance}). Clamping.")
                    distance = max_distance
                min_scores[ref_ann["id"]] = min(min_scores[ref_ann["id"]], distance)

            d_e_values.extend(min_scores.values())

    return {'d_o': d_o_values, 'd_e': d_e_values}


# --- 2. Main Orchestration ---

def calculate_empirical_disagreement(cfg: EmpiricalDisagreementConfig):
    """
    API-ready orchestration for empirical disagreement calculation.

    Loads data, applies the specified similarity function, calculates the 
    observed and expected disagreement distributions, and exports the results.

    Args:
        cfg (EmpiricalDisagreementConfig): Configuration object containing 
            paths and parameters for generating D_o and D_e distributions.
    """
    setup_kalos_logging(cfg.log_level)
    
    try:
        similarity_function = SIMILARITY_FUNCTIONS[cfg.similarity_func]
    except KeyError:
        logger.error(f"Invalid similarity function '{cfg.similarity_func}'. Available: {list(SIMILARITY_FUNCTIONS.keys())}")
        return

    coco_data = load_annotations(str(cfg.annotation_file))
    processed_data = preprocess_data(coco_data)

    if cfg.only_with_annotations:
        original_count = len(processed_data)
        processed_data = {
            img_id: data for img_id, data in processed_data.items()
            if any(len(anns) > 0 for anns in data['annotations_by_rater'].values())
        }
        logger.info(f"Filtered out {original_count - len(processed_data)} images. Remaining: {len(processed_data)}")

    logger.info(f"Calculating empirical disagreement using similarity function: '{cfg.similarity_func}'")
    results = calculate_do_de(processed_data, similarity_function)

    with open(str(cfg.output_file), 'w') as f:
        json.dump(results, f, indent=4)

    logger.info(f"Successfully calculated D_o and D_e.")
    logger.info(f"   - D_o values calculated: {len(results['d_o'])}")
    logger.info(f"   - D_e values calculated: {len(results['d_e'])}")
    logger.info(f"   - Results saved to: {cfg.output_file}")
