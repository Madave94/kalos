import matplotlib.pyplot as plt
import numpy as np
import logging
from kalos.utils.theme_manager import theme_manager, PROJECT_COLORS_HEX

logger = logging.getLogger(__name__)

def plot_localization_sensitivity(lsa_mean, lsa_global, output_file=None, file_format='png'):
    """
    Plots the localization sensitivity analysis (Mean vs Global Alpha).
    """
    scale = theme_manager.font_scale
    
    # 1. Align and sort results by threshold
    sorted_thresholds = sorted(lsa_mean.keys())
    mean_alphas = [lsa_mean[t] for t in sorted_thresholds]
    global_alphas = [lsa_global.get(t, np.nan) for t in sorted_thresholds]

    fig, ax = plt.subplots(figsize=(10, 6))

    # 2. Plot Mean Alpha (Primary)
    ax.plot(sorted_thresholds, mean_alphas,
            marker='o',
            linestyle='-',
            color=PROJECT_COLORS_HEX['PRIMARY'],
            linewidth=2 * scale,
            markersize=8 * scale,
            label="Mean Image K-α")

    # 3. Plot Global Alpha (Accent)
    ax.plot(sorted_thresholds, global_alphas,
            marker='s',
            linestyle='--',
            color=PROJECT_COLORS_HEX['ACCENT'],
            linewidth=2 * scale,
            markersize=8 * scale,
            label="Global Dataset K-α")

    # Aesthetics
    ax.set_xlabel("Similarity Threshold")
    ax.set_ylabel("K-α")
    ax.grid(True, linestyle='--', alpha=0.7)

    ax.set_xticks(sorted_thresholds)
    
    # Combine all valid values for limits
    all_vals = [v for v in mean_alphas + global_alphas if not np.isnan(v)]
    if all_vals:
        ax.set_ylim(min(all_vals) - 0.1, max(all_vals) + 0.1)
    
    ax.set_xlim(min(sorted_thresholds) - 0.05, max(sorted_thresholds) + 0.05)

    ax.legend(loc='lower left', fontsize=12 * scale)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.15)

    if output_file:
        plt.savefig(output_file, dpi=300 if file_format == 'png' else None, bbox_inches='tight')
    else:
        plt.show()
