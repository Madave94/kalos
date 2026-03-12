"""
Pipeline for deriving principled localization thresholds using KS-statistics.
Analyzes disagreement distributions to identify the optimal boundary between
valid correspondences (signal) and random overlap (noise).
"""

import os
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from typing import List, Optional, Dict

from kalos.utils.theme_manager import theme_manager, PROJECT_COLORS_HEX
from kalos.utils.logging import setup_kalos_logging
from kalos.config import PlottingConfig

logger = logging.getLogger(__name__)

def find_bayesian_boundary(d_o: np.ndarray, d_e: np.ndarray) -> float:
    """
    Calculates the optimal localization threshold tau* using PDF intersection.

    Identifies the boundary between Signal (D_o) and Noise (D_e) distributions 
    using Bayesian Decision Boundary logic derived from Gaussian KDE.

    Args:
        d_o (np.ndarray): Array of observed disagreement distances.
        d_e (np.ndarray): Array of expected disagreement distances.

    Returns:
        float: The optimal distance threshold tau*. Defaults to 0.5 on failure.
    """
    if len(d_o) < 2 or len(d_e) < 2:
        return 0.5
    
    try:
        kde_do = gaussian_kde(d_o)
        kde_de = gaussian_kde(d_e)
    except Exception:
        return 0.5

    x_grid = np.linspace(0, 1, 1000)
    pdf_do = kde_do(x_grid)
    pdf_de = kde_de(x_grid)

    mode_do_idx = np.argmax(pdf_do)
    mode_do = x_grid[mode_do_idx]
    
    diff = pdf_do - pdf_de
    signs = np.sign(diff)
    sign_changes = ((np.roll(signs, -1) - signs) != 0).astype(int)
    sign_changes[-1] = 0
    intersection_indices = np.where(sign_changes == 1)[0]
    
    valid_indices = [idx for idx in intersection_indices if x_grid[idx] > mode_do]
    
    if valid_indices:
        best_idx = valid_indices[0]
        return x_grid[best_idx]
    
    mode_de_idx = np.argmax(pdf_de)
    start_idx = min(mode_do_idx, mode_de_idx)
    end_idx = max(mode_do_idx, mode_de_idx)
    
    if start_idx == end_idx:
        return 0.5
        
    combined_pdf = pdf_do + pdf_de
    valley_pdf = combined_pdf[start_idx:end_idx+1]
    min_local_idx = np.argmin(valley_pdf)
    min_density_idx = start_idx + min_local_idx
    
    return x_grid[min_density_idx]

def calculate_ks_statistic(d_o: np.ndarray, d_e: np.ndarray) -> float:
    """
    Calculates the Kolmogorov-Smirnov statistic for two distributions.

    Args:
        d_o (np.ndarray): Array of observed disagreement distances.
        d_e (np.ndarray): Array of expected disagreement distances.

    Returns:
        float: The maximum difference between the two cumulative distributions.

    Raises:
        ValueError: If either input array is empty.
    """
    if len(d_o) == 0 or len(d_e) == 0:
        raise ValueError("Input arrays cannot be empty.")
        
    d_o_sorted = np.sort(d_o)
    d_e_sorted = np.sort(d_e)
    all_values = np.unique(np.concatenate([d_o_sorted, d_e_sorted]))
    cdf_do = np.searchsorted(d_o_sorted, all_values, side='right') / len(d_o_sorted)
    cdf_de = np.searchsorted(d_e_sorted, all_values, side='right') / len(d_e_sorted)
    return np.max(cdf_do - cdf_de)

def plot_disagreement_distributions(
    d_o: np.ndarray, 
    d_e: np.ndarray, 
    metric_name: str, 
    ks_statistic: float, 
    output_dir: str = "plots", 
    tau_star: Optional[float] = None
):
    """
    Plots the relative frequency histograms for D_o and D_e.

    Args:
        d_o (np.ndarray): Observed distances.
        d_e (np.ndarray): Expected distances.
        metric_name (str): Name of the similarity metric used.
        ks_statistic (float): Calculated KS score for reporting.
        output_dir (str): Directory to save the resulting PDF.
        tau_star (float, optional): Optimal threshold to indicate with a vertical line.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    scale = theme_manager.font_scale
    fig, ax = plt.subplots(figsize=(12, 7))

    all_data = np.concatenate([d_o, d_e])
    bins = np.linspace(min(all_data), max(all_data), 50)

    weights_do = np.ones_like(d_o) / len(d_o)
    ax.hist(d_o, bins=bins, weights=weights_do, color=PROJECT_COLORS_HEX['PRIMARY'], alpha=0.7, label='Observed Distance ($D_o$)')
    mean_do = np.mean(d_o)
    ax.axvline(mean_do, color=PROJECT_COLORS_HEX['PRIMARY'], linestyle='dashed', linewidth=2 * scale, label=f'Mean $D_o$: {mean_do:.2f}')

    weights_de = np.ones_like(d_e) / len(d_e)
    ax.hist(d_e, bins=bins, weights=weights_de, color=PROJECT_COLORS_HEX['ACCENT'], alpha=0.7, label='Expected Distance ($D_e$)')
    mean_de = np.mean(d_e)
    ax.axvline(mean_de, color=PROJECT_COLORS_HEX['ACCENT'], linestyle='dashed', linewidth=2 * scale, label=f'Mean $D_e$: {mean_de:.2f}')

    if tau_star is not None:
            ax.axvline(tau_star, color=PROJECT_COLORS_HEX['OKAY_ACCENT'], linestyle='dotted', linewidth=2 * scale, label=f'Optimal Threshold $\\tau^*$: {tau_star:.3f}')

    ax.set_xlabel('Distance Value ($1 - Similarity$)')
    ax.set_ylabel('Relative Frequency')
    
    # Use a very subtle margin to ensure boundary bars (0.0 and 1.0) are fully rendered 
    # and not visually clipped by the axes lines.
    ax.set_xlim(-0.01, 1.01)
    
    # Precise Ticks: 0.0, 0.2, 0.4, 0.6, 0.8, 1.0
    ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    ax.set_xticks(ticks)
    
    # Create a twin axis to show similarity
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(ticks)
    # Mirror labels: 0.0 dist -> 1.0 sim, 1.0 dist -> 0.0 sim
    ax2.set_xticklabels([f"{1-x:.1f}" for x in ticks])
    ax2.set_xlabel('Equivalent Similarity Score')

    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{metric_name}_distribution.pdf'), bbox_inches='tight')
    plt.close()

def derive_principled_configuration(
    disagreement_files: List[str],
    output_dir: str = "plots",
    plotting: PlottingConfig = None,
    log_level: str = "INFO"
):
    """
    API-ready orchestration for deriving principled configuration.

    Analyzes multiple disagreement files to find the best metric and threshold.

    Args:
        disagreement_files (List[str]): List of paths to D_o/D_e JSON files.
        output_dir (str): Directory where principled plots will be saved.
        plotting (PlottingConfig, optional): Central plotting configuration.
        log_level (str, optional): Logging verbosity. Defaults to "INFO".
    """
    setup_kalos_logging(log_level)
    
    # Initialize styling from central config
    if plotting:
        theme_manager.apply(
            theme_name=plotting.theme,
            font_family=plotting.font_family,
            font_scale=plotting.font_scale,
            overrides=plotting.color_overrides
        )
    
    results = {}
    tau_results = {}

    for file_path in disagreement_files:
        if not os.path.exists(file_path):
            logger.warning(f"File not found, skipping: {file_path}")
            continue
        if not file_path.endswith('.json'):
            logger.warning(f"File is not a JSON file, skipping: {file_path}")
            continue
            
        try:
            metric_name = os.path.splitext(os.path.basename(file_path))[0].replace('_disagreements', '')
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            d_o = np.array([x for x in data['d_o'] if x is not None])
            d_e = np.array([x for x in data['d_e'] if x is not None])

            # Validation: Ensure all distances are in [0, 1]
            if np.any(d_o < 0.0) or np.any(d_o > 1.0) or np.any(d_e < 0.0) or np.any(d_e > 1.0):
                logger.error(f"Validation failed for {file_path}: Disagreement values must be in range [0.0, 1.0].")
                continue

            ks_statistic = calculate_ks_statistic(d_o, d_e)
            tau_star = find_bayesian_boundary(d_o, d_e)

            results[metric_name] = ks_statistic
            tau_results[metric_name] = tau_star

            plot_disagreement_distributions(d_o, d_e, metric_name, ks_statistic, output_dir, tau_star)
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            logger.error(f"Skipping {file_path} due to error: {e}")
            continue

    if not results:
        logger.warning("No valid disagreement files to process.")
        return

    best_metric = max(results, key=results.get)
    
    logger.info("--- KS Statistics Results ---")
    for metric, score in sorted(results.items(), key=lambda item: item[1], reverse=True):
        logger.info(f"{metric}: KS={score:.4f}, Tau*={tau_results[metric]:.4f}")
    
    logger.info(f"The best distance metric is: '{best_metric}' with a KS score of {results[best_metric]:.4f} and Tau*={tau_results[best_metric]:.4f}")

    logger.info("--- Interpretation & Next Steps ---")
    logger.info(f"The optimal disagreement distance (Tau*) for '{best_metric}' is {tau_results[best_metric]:.4f}.")
    logger.info("This value represents the boundary between 'Signal' (valid matches) and 'Noise' (random chance).")
    logger.info(f"  -> Objects with Distance <= {tau_results[best_metric]:.4f} are considered valid correspondences.")
    logger.info(f"  -> This corresponds to a Similarity Score >= {1 - tau_results[best_metric]:.4f}.")
    
    logger.info("Usage in kalos_execution.py:")
    logger.info(f"    -> Use --similarity_threshold {1 - tau_results[best_metric]:.4f}  (calculated as 1 - Tau*)")
    logger.info("  Note: kalos_execution.py expects a Similarity threshold (Higher is Better).")
