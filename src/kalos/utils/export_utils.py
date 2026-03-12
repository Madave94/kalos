"""
Data export adapters for the KaLOS toolkit.
Handles the serialization of mathematically derived agreement metrics into 
human-readable CSV reports and a comprehensive JSON checkpoint for decoupled plotting.
"""

import csv
import json
import logging
import os
import numpy as np
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

def export_iaa_results(
    output_dir: str,
    mean_alpha: float,
    global_alpha: float,
    mean_vitalities: Dict[str, List[float]],
    global_vitalities: Dict[str, float],
    mean_difficulties: Dict[str, Dict[str, Any]],
    global_difficulties: Dict[str, Dict[str, Any]],
    image_alphas: Dict[int, float],
    mean_collaboration_matrix: Dict[str, Dict[str, List[float]]],
    global_collaboration_matrix: Dict[str, Dict[str, float]],
    intra_iaa_results: Dict[str, float],
    session_iaa_results: Dict[str, float],
    lsa_mean: Dict[float, float],
    lsa_global: Dict[float, float],
    task: str,
    similarity_threshold: float,
    all_raters: List[str]
):
    """
    Exports IAA calculation results to CSV and JSON files, including a checkpoint for plotting.

    Args:
        output_dir (str): Directory where the output files will be saved.
        mean_alpha (float): The primary Mean Image K-Alpha score.
        global_alpha (float): The secondary Global Dataset K-Alpha score.
        mean_vitalities (Dict[str, List[float]]): Annotator vitality data (Mean impact).
        global_vitalities (Dict[str, float]): Annotator vitality data (Global impact).
        mean_difficulties (Dict[str, Dict[str, Any]]): Class difficulty data (Mean impact).
        global_difficulties (Dict[str, Dict[str, Any]]): Class difficulty data (Global impact).
        image_alphas (Dict[int, float]): Dictionary mapping image IDs to their Alpha scores.
        mean_collaboration_matrix (Dict[str, Dict[str, List[float]]]): Pairwise agreement (Mean).
        global_collaboration_matrix (Dict[str, Dict[str, float]]): Pairwise agreement (Global).
        intra_iaa_results (Dict[str, float]): Intra-annotator self-consistency scores.
        session_iaa_results (Dict[str, float]): Per-session agreement scores.
        lsa_mean (Dict[float, float]): Localization sensitivity analysis (Mean scores).
        lsa_global (Dict[float, float]): Localization sensitivity analysis (Global scores).
        task (str): The name of the annotation task (e.g., 'bbox', 'segm').
        similarity_threshold (float): The base similarity threshold used for matching.
        all_raters (List[str]): List of all unique rater identities in the dataset.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Summary CSV
    summary_path = os.path.join(output_dir, "iaa_summary.csv")
    with open(summary_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Task", task])
        writer.writerow(["Similarity Threshold", similarity_threshold])
        writer.writerow(["Mean Image K-Alpha (Primary)", f"{mean_alpha:.4f}"])
        writer.writerow(["Global Dataset K-Alpha (Secondary)", f"{global_alpha:.4f}"])
    logger.info(f"Summary results exported to: {summary_path}")

    # 2. Class Difficulties CSV
    if mean_difficulties:
        class_path = os.path.join(output_dir, "class_difficulties.csv")
        with open(class_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Class Name", "Mean Image K-Alpha", "Global Dataset K-Alpha"])
            for class_name in sorted(mean_difficulties.keys()):
                avg_alpha = np.mean(mean_difficulties[class_name]["alphas"])
                g_alpha = global_difficulties.get(class_name, {}).get("alpha", 0.0)
                writer.writerow([class_name, f"{avg_alpha:.4f}", f"{g_alpha:.4f}"])
        logger.info(f"Class difficulty results exported to: {class_path}")

    # 3. Annotator Vitality CSV
    if mean_vitalities:
        vitality_path = os.path.join(output_dir, "annotator_vitality.csv")
        with open(vitality_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Annotator ID", "Mean Vitality", "Global Vitality"])
            for rater_id in sorted(mean_vitalities.keys()):
                avg_vitality = np.mean(mean_vitalities[rater_id])
                g_vitality = global_vitalities.get(rater_id, 0.0)
                writer.writerow([rater_id, f"{avg_vitality:.4f}", f"{g_vitality:.4f}"])
        logger.info(f"Annotator vitality results exported to: {vitality_path}")

    # 4. Intra-IAA Consistency CSV
    if intra_iaa_results:
        intra_path = os.path.join(output_dir, "intra_iaa_consistency.csv")
        with open(intra_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Annotator ID", "Self-Consistency (Mean Image K-Alpha)"])
            for rater_id, score in sorted(intra_iaa_results.items()):
                writer.writerow([rater_id, f"{score:.4f}"])
        logger.info(f"Intra-IAA consistency results exported to: {intra_path}")

    # 5. Per-Session Performance CSV
    if session_iaa_results:
        session_path = os.path.join(output_dir, "per_session_performance.csv")
        with open(session_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Session ID", "Mean Inter-IAA K-Alpha"])
            for s_id, score in sorted(session_iaa_results.items()):
                writer.writerow([s_id, f"{score:.4f}"])
        logger.info(f"Per-session results exported to: {session_path}")

    # 6. Localization Sensitivity CSV
    if lsa_mean:
        lsa_path = os.path.join(output_dir, "localization_sensitivity.csv")
        with open(lsa_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Threshold", "Mean Image K-Alpha", "Global Dataset K-Alpha"])
            for thresh in sorted(lsa_mean.keys()):
                m_score = lsa_mean[thresh]
                g_score = lsa_global.get(thresh, np.nan)
                writer.writerow([f"{thresh:.4f}", f"{m_score:.4f}", f"{g_score:.4f}"])
        logger.info(f"LSA results exported to: {lsa_path}")

    # 7. Detailed Per-Image Alphas JSON
    detailed_path = os.path.join(output_dir, "per_image_alphas.json")
    with open(detailed_path, 'w') as f:
        # Convert keys to strings for JSON serialization
        serializable_image_alphas = {str(k): v for k, v in image_alphas.items()}
        json.dump(serializable_image_alphas, f, indent=4)
    logger.info(f"Detailed image alphas exported to: {detailed_path}")

    # 8. Collaboration Matrix CSV
    if mean_collaboration_matrix or global_collaboration_matrix:
        collab_path = os.path.join(output_dir, "collaboration_matrix.csv")
        sorted_raters = sorted(all_raters)

        with open(collab_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            # Two tables: Mean and Global
            writer.writerow(["--- MEAN PAIRWISE ALPHA ---"])
            writer.writerow([""] + sorted_raters)
            for r1 in sorted_raters:
                row = [r1]
                for r2 in sorted_raters:
                    if r1 == r2:
                        row.append("1.0000")
                    else:
                        sorted_pair = tuple(sorted((r1, r2)))
                        alphas = mean_collaboration_matrix.get(sorted_pair[0], {}).get(sorted_pair[1], [])
                        if not alphas:
                            row.append("NaN")
                        else:
                            row.append(f"{np.mean(alphas):.4f}")
                writer.writerow(row)
            
            writer.writerow([])
            writer.writerow(["--- GLOBAL PAIRWISE ALPHA ---"])
            writer.writerow([""] + sorted_raters)
            for r1 in sorted_raters:
                row = [r1]
                for r2 in sorted_raters:
                    if r1 == r2:
                        row.append("1.0000")
                    else:
                        val = global_collaboration_matrix.get(r1, {}).get(r2, np.nan)
                        row.append(f"{val:.4f}" if not np.isnan(val) else "NaN")
                writer.writerow(row)
        logger.info(f"Collaboration matrices exported to: {collab_path}")

    # 9. Plotting Checkpoint JSON
    checkpoint_path = os.path.join(output_dir, "kalos_checkpoint.json")
    checkpoint_data = {
        "metadata": {
            "task": task,
            "similarity_threshold": similarity_threshold,
            "all_raters": all_raters
        },
        "results": {
            "mean_alpha": mean_alpha,
            "global_alpha": global_alpha,
            "mean_vitalities": mean_vitalities,
            "global_vitalities": global_vitalities,
            "mean_difficulties": mean_difficulties,
            "global_difficulties": global_difficulties,
            "image_alphas": {str(k): v for k, v in image_alphas.items()},
            "mean_collaboration_matrix": mean_collaboration_matrix,
            "global_collaboration_matrix": global_collaboration_matrix,
            "intra_iaa_results": intra_iaa_results,
            "session_iaa_results": session_iaa_results,
            "lsa_mean": {str(k): v for k, v in lsa_mean.items()},
            "lsa_global": {str(k): v for k, v in lsa_global.items()}
        }
    }
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint_data, f, indent=4)
    logger.info(f"Plotting checkpoint exported to: {checkpoint_path}")
