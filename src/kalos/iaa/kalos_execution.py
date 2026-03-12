"""
Orchestrates the execution of the KaLOS mathematical pipeline.
Responsible for data loading, pre-processing, metric calculation across 
all three tiers of agreement, and exporting the final serialized results.
"""

import logging
from typing import Dict, List, Any, Optional
import numpy as np

from kalos.correspondence import correspondence_algorithms
from kalos.iaa.core import calculate_iaa
from kalos.config import KaLOSProjectConfig
from kalos.utils.logging import setup_kalos_logging
from kalos.utils.export_utils import export_iaa_results

logger = logging.getLogger(__name__)

# --- 1. Data Layer Orchestration ---
def load_and_preprocess_data(annotation_file: str, annotation_type: str):
    """Loads and preprocesses data based on annotation type. Decoupled from Config object."""
    if annotation_type == 'coco-json':
        coco_data = correspondence_algorithms.load_annotations(annotation_file)
        categories = {cat["id"]: cat["name"] for cat in coco_data["categories"]}
        processed_data = correspondence_algorithms.preprocess_data(coco_data)
        
        # Derive all_raters from the processed data to capture session-aware identities
        all_raters_set = set()
        for img_data in processed_data.values():
            all_raters_set.update(img_data['rater_list'])
        all_raters = sorted(list(all_raters_set))
        
        return {
            "processed_data": processed_data,
            "categories": categories,
            "all_raters": all_raters
        }
    elif annotation_type == 'lidc-idri-json':
        annotation_data = correspondence_algorithms.load_annotations(annotation_file)
        processed_data = correspondence_algorithms.preprocess_data(annotation_data)
        
        # Consistent derivation for LIDC
        all_raters_set = set()
        for img_data in processed_data.values():
            all_raters_set.update(img_data['rater_list'])
        all_raters = sorted(list(all_raters_set))

        return {
            "processed_data": processed_data,
            "categories": {},
            "all_raters": all_raters
        }
    else:
        raise ValueError(f"Unsupported annotation type: {annotation_type}")


# --- 2. Main Execution Orchestrator ---
def run_kalos_pipeline(cfg: KaLOSProjectConfig):
    """
    Orchestrates the main mathematical evaluation pipeline for KaLOS.

    Executes data loading, applies correspondence matching, calculates the 
    3-Tier reliability metrics (Overall, Per-Session, Intra-Annotator), 
    and handles result serialization for downstream plotting.

    Args:
        cfg (KaLOSProjectConfig): The master configuration object defining 
            the task parameters and output settings.
    """
    setup_kalos_logging(cfg.log_level)
    
    # Load and preprocess data
    data_package = load_and_preprocess_data(str(cfg.annotation_file), cfg.annotation_type)
    processed_data = data_package["processed_data"]
    categories = data_package["categories"]
    all_raters = data_package.get("all_raters")

    if cfg.only_with_instances:
        original_count = len(processed_data)
        processed_data = {
            img_id: img_data 
            for img_id, img_data in processed_data.items() 
            if any(anns for anns in img_data['annotations_by_rater'].values())
        }
        filtered_count = len(processed_data)
        if original_count != filtered_count:
            logger.info(f"Filtering enabled: Running on {filtered_count} images containing instances (Excluded {original_count - filtered_count} empty images).")

    logger.info("--- Correspondence Configuration ---")
    logger.info(f"Method: '{cfg.method}'")
    logger.info(f"Similarity Function: '{cfg.threshold_func}'")
    logger.info(f"Cost Function: '{cfg.cost_func}'")
    logger.info(f"Similarity Threshold: {cfg.similarity_threshold:.2f} (Accepting matches with Similarity >= {cfg.similarity_threshold:.2f})")

    # Call the core calculation logic
    (mean_alpha, global_alpha, mean_vitalities, global_vitalities, 
     mean_difficulties, global_difficulties, image_alphas, 
     mean_collaboration_matrix, global_collaboration_matrix) = calculate_iaa(
        processed_data=processed_data,
        categories=categories,
        method=cfg.method,
        threshold_func=cfg.threshold_func,
        cost_func=cfg.cost_func,
        similarity_threshold=cfg.similarity_threshold,
        calculate_vitality=cfg.calculate_vitality,
        calculate_difficulty=cfg.calculate_difficulty,
        collaboration_clusters=cfg.collaboration_clusters,
        all_raters=all_raters
    )

    logger.info(f"\nKrippendorff's Alpha Evaluation ({cfg.task}) at base similarity threshold {cfg.similarity_threshold}:")
    logger.info(f"  - Mean Image Alpha (Primary): {mean_alpha:.4f}")
    logger.info(f"  - Global Dataset Alpha (Secondary): {global_alpha:.4f}")
    logger.info("  *Note: Mean Image Alpha is the empirically validated metric for this toolkit.")

    # --- Downstream Analysis Display ---
    if cfg.calculate_difficulty:
        logger.info("\n--- Class Recognition Difficulty ---")
        for class_name, data in sorted(mean_difficulties.items()):
            avg_alpha = np.mean(data["alphas"])
            global_alpha_cls = global_difficulties.get(class_name, {}).get("alpha", 0.0)
            logger.info(f"  Class: {class_name:15} | Mean Alpha: {avg_alpha:.4f} | Global Alpha: {global_alpha_cls:.4f}")

    if cfg.calculate_vitality:
        logger.info("\n--- Annotator Vitality ---")
        if not mean_vitalities:
            logger.info("No images with 3 or more annotators found. Vitality not calculated.")
        else:
            for rater_id, vitalities in sorted(mean_vitalities.items()):
                avg_vitality = np.mean(vitalities)
                g_vitality = global_vitalities.get(rater_id, 0.0)
                logger.info(f"  - Annotator {rater_id:15}: Mean Vitality = {avg_vitality:.4f} | Global Vitality = {g_vitality:.4f}")

    # --- Localization Sensitivity Analysis ---
    lsa_mean = {}
    lsa_global = {}
    if cfg.localization_sensitivity_thresholds:
        logger.info(f"Running Localization Sensitivity Analysis on {len(cfg.localization_sensitivity_thresholds)} thresholds.")
        lsa_mean = {cfg.similarity_threshold: mean_alpha}
        lsa_global = {cfg.similarity_threshold: global_alpha}

        for threshold in sorted(cfg.localization_sensitivity_thresholds):
            if threshold == cfg.similarity_threshold: continue
            m_alpha, g_alpha, _, _, _, _, _, _, _ = calculate_iaa(
                processed_data=processed_data,
                categories=categories,
                method=cfg.method,
                threshold_func=cfg.threshold_func,
                cost_func=cfg.cost_func,
                similarity_threshold=threshold,
                calculate_vitality=False,
                calculate_difficulty=False,
                collaboration_clusters=False,
                all_raters=all_raters
            )
            lsa_mean[threshold] = m_alpha
            lsa_global[threshold] = g_alpha

    if cfg.collaboration_clusters:
        logger.info("\n--- Collaboration Cluster Analysis Report ---")

    # --- 4. Intra-Annotator Agreement (Consistency) ---
    intra_iaa_results = {}
    session_iaa_results = {}
    
    if cfg.calculate_intra_iaa:
        # TIER 2: Per-Session Inter-Annotator Agreement (Team performance per time-step)
        logger.info("\n--- Per-Session Inter-Annotator Agreement (Team Performance) ---")
        unique_sessions = sorted(list(set(r.split(" (S")[1].rstrip(")") for r in all_raters if " (S" in r)))
        
        for s_id in unique_sessions:
            session_suffix = f" (S{s_id})"
            session_raters = [r for r in all_raters if r.endswith(session_suffix)]
            
            if len(session_raters) < 2:
                continue
                
            session_processed_data = {}
            for img_id, img_data in processed_data.items():
                img_session_raters = [r for r in img_data['rater_list'] if r in session_raters]
                if len(img_session_raters) >= 2:
                    session_processed_data[img_id] = {
                        'rater_list': img_session_raters,
                        'annotations_by_rater': {r: img_data['annotations_by_rater'][r] for r in img_session_raters}
                    }
            
            if session_processed_data:
                m_alpha, g_alpha, _, _, _, _, _, _, _ = calculate_iaa(
                    processed_data=session_processed_data,
                    categories=categories,
                    method=cfg.method,
                    threshold_func=cfg.threshold_func,
                    cost_func=cfg.cost_func,
                    similarity_threshold=cfg.similarity_threshold,
                    calculate_vitality=False,
                    calculate_difficulty=False,
                    collaboration_clusters=False,
                    all_raters=session_raters
                )
                session_iaa_results[s_id] = m_alpha
                logger.info(f"  Session {s_id}: Mean Inter-IAA = {m_alpha:.4f}")

        # TIER 3: Intra-Annotator Agreement (Self-Consistency)
        logger.info("\n--- Intra-Annotator Agreement (Self-Consistency) ---")
        unique_raters = sorted(list(set(r.split(" (S")[0] for r in all_raters)))
        
        for rater_id in unique_raters:
            rater_identities = [r for r in all_raters if r.startswith(rater_id + " (S") or r == rater_id]
            if len(rater_identities) < 2:
                continue
                
            intra_processed_data = {}
            for img_id, img_data in processed_data.items():
                img_rater_sessions = [r for r in img_data['rater_list'] if r in rater_identities]
                if len(img_rater_sessions) >= 2:
                    intra_processed_data[img_id] = {
                        'rater_list': img_rater_sessions,
                        'annotations_by_rater': {r: img_data['annotations_by_rater'][r] for r in img_rater_sessions}
                    }
            
            if intra_processed_data:
                m_alpha, g_alpha, _, _, _, _, _, _, _ = calculate_iaa(
                    processed_data=intra_processed_data,
                    categories=categories,
                    method=cfg.method,
                    threshold_func=cfg.threshold_func,
                    cost_func=cfg.cost_func,
                    similarity_threshold=cfg.similarity_threshold,
                    calculate_vitality=False,
                    calculate_difficulty=False,
                    collaboration_clusters=False,
                    all_raters=rater_identities
                )
                intra_iaa_results[rater_id] = m_alpha
                logger.info(f"  Annotator {rater_id}: Mean consistency = {m_alpha:.4f}")

    # --- 5. Export Results ---
    if cfg.output_results:
        export_iaa_results(
            output_dir=str(cfg.output_results),
            mean_alpha=mean_alpha,
            global_alpha=global_alpha,
            mean_vitalities=mean_vitalities,
            global_vitalities=global_vitalities,
            mean_difficulties=mean_difficulties,
            global_difficulties=global_difficulties,
            image_alphas=image_alphas,
            mean_collaboration_matrix=mean_collaboration_matrix,
            global_collaboration_matrix=global_collaboration_matrix,
            intra_iaa_results=intra_iaa_results,
            session_iaa_results=session_iaa_results,
            lsa_mean=lsa_mean,
            lsa_global=lsa_global,
            task=cfg.task,
            similarity_threshold=cfg.similarity_threshold,
            all_raters=all_raters
        )
