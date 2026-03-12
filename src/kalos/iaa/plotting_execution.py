"""
Orchestrates the diagnostic plotting for KaLOS.
Loads serialized results from a checkpoint and generates granular, 
theme-aware visualizations without re-running the mathematical pipeline.
"""

import json
import logging
import os

from kalos.config import KaLOSProjectConfig
from kalos.utils.logging import setup_kalos_logging
from kalos.utils.theme_manager import theme_manager

from kalos.diagnostics.per_image_distribution_plot import plot_alpha_distribution
from kalos.diagnostics.localization_sensitivity_plot import plot_localization_sensitivity
from kalos.diagnostics.heatmap_collaboration_cluster import plot_collaboration_heatmap
from kalos.diagnostics.annotator_vitality_plot import plot_annotator_vitality
from kalos.diagnostics.class_recognition_difficulty_plot import plot_class_difficulty

logger = logging.getLogger(__name__)

def run_plotting_pipeline(cfg: KaLOSProjectConfig):
    """
    Orchestrates diagnostic plotting from a serialized results checkpoint.
    Uses the master configuration flags to determine which plots should be generated.

    Args:
        cfg (KaLOSProjectConfig): The master project configuration object.
    """
    setup_kalos_logging(cfg.log_level)
    
    if not cfg.output_results:
        raise ValueError("Field 'output_results' is required to locate the results checkpoint for plotting.")
    
    # 1. Load Checkpoint
    checkpoint_path = os.path.join(str(cfg.output_results), "kalos_checkpoint.json")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}. Run 'execute' first.")
    
    logger.info(f"Loading results checkpoint from: {checkpoint_path}")
    with open(checkpoint_path, 'r') as f:
        checkpoint_data = json.load(f)
    
    results = checkpoint_data["results"]
    metadata = checkpoint_data["metadata"]
    all_raters = metadata["all_raters"]
    plot_fmt = cfg.plotting.plot_format
    output_folder = cfg.plotting.output_folder or str(cfg.output_results)
    os.makedirs(output_folder, exist_ok=True)

    # 2. Diagnostic Plotting Tiers
    
    # TIER 1: Alpha Distribution (Always enabled by default, gated by plot setting)
    s = cfg.plotting.alpha_distribution
    if s.enabled or cfg.plotting.plot_all:
        theme_manager.apply(
            theme_name=s.theme or cfg.plotting.theme,
            font_family=s.font_family or cfg.plotting.font_family,
            font_scale=s.font_scale or cfg.plotting.font_scale,
            overrides=s.color_overrides or cfg.plotting.color_overrides
        )
        path = s.output_path or os.path.join(output_folder, f"alpha_distribution.{plot_fmt}")
        plot_alpha_distribution(results["image_alphas"], results["global_alpha"], output_file=path, file_format=plot_fmt)

    # TIER 2: Collaboration Heatmap (Mean and Global)
    s = cfg.plotting.collaboration_heatmap
    if cfg.collaboration_clusters and (s.enabled or cfg.plotting.plot_all):
        theme_manager.apply(
            theme_name=s.theme or cfg.plotting.theme,
            font_family=s.font_family or cfg.plotting.font_family,
            font_scale=s.font_scale or cfg.plotting.font_scale,
            overrides=s.color_overrides or cfg.plotting.color_overrides
        )
        # Mean Heatmap
        path_mean = s.output_path or os.path.join(output_folder, f"collaboration_heatmap_mean.{plot_fmt}")
        plot_collaboration_heatmap(results["mean_collaboration_matrix"], all_raters, output_file=path_mean, file_format=plot_fmt, label="Mean Pairwise K-α")
        # Global Heatmap
        path_global = s.output_path or os.path.join(output_folder, f"collaboration_heatmap_global.{plot_fmt}")
        plot_collaboration_heatmap(results["global_collaboration_matrix"], all_raters, output_file=path_global, file_format=plot_fmt, label="Global Pairwise K-α")

    # TIER 3: Localization Sensitivity (LSA)
    s = cfg.plotting.localization_sensitivity
    if cfg.localization_sensitivity_thresholds and (s.enabled or cfg.plotting.plot_all):
        theme_manager.apply(
            theme_name=s.theme or cfg.plotting.theme,
            font_family=s.font_family or cfg.plotting.font_family,
            font_scale=s.font_scale or cfg.plotting.font_scale,
            overrides=s.color_overrides or cfg.plotting.color_overrides
        )
        # JSON keys are strings, convert back to float
        lsa_mean_converted = {float(k): v for k, v in results["lsa_mean"].items()}
        lsa_global_converted = {float(k): v for k, v in results["lsa_global"].items()}
        
        path = s.output_path or os.path.join(output_folder, f"localization_sensitivity.{plot_fmt}")
        plot_localization_sensitivity(lsa_mean_converted, lsa_global_converted, output_file=path, file_format=plot_fmt)

    # TIER 4: Annotator Vitality
    s = cfg.plotting.annotator_vitality
    if cfg.calculate_vitality and (s.enabled or cfg.plotting.plot_all):
        theme_manager.apply(
            theme_name=s.theme or cfg.plotting.theme,
            font_family=s.font_family or cfg.plotting.font_family,
            font_scale=s.font_scale or cfg.plotting.font_scale,
            overrides=s.color_overrides or cfg.plotting.color_overrides
        )
        path = s.output_path or os.path.join(output_folder, f"annotator_vitality.{plot_fmt}")
        plot_annotator_vitality(results["mean_vitalities"], output_file=path, file_format=plot_fmt)
    
    # TIER 5: Class Difficulty
    s = cfg.plotting.class_difficulty
    if cfg.calculate_difficulty and (s.enabled or cfg.plotting.plot_all):
        theme_manager.apply(
            theme_name=s.theme or cfg.plotting.theme,
            font_family=s.font_family or cfg.plotting.font_family,
            font_scale=s.font_scale or cfg.plotting.font_scale,
            overrides=s.color_overrides or cfg.plotting.color_overrides
        )
        path = s.output_path or os.path.join(output_folder, f"class_difficulty.{plot_fmt}")
        plot_class_difficulty(results["mean_difficulties"], results["global_difficulties"], output_file=path, file_format=plot_fmt)

    logger.info("✅ Plotting pipeline complete.")