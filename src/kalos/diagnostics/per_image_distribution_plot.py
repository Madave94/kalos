import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter
from kalos.utils.theme_manager import theme_manager, PROJECT_COLORS_HEX

def plot_alpha_distribution(image_alphas, global_alpha, output_file=None, file_format='png'):
    """
    Plots the distribution of Krippendorff's alpha values per image.
    """
    scale = theme_manager.font_scale
    alpha_values = list(image_alphas.values())
    mean_alpha = np.mean(alpha_values)
    median_alpha = np.median(alpha_values)

    fig, ax1 = plt.subplots(figsize=(10, 4.5))

    # --- Primary Y-Axis (Linear Scale) ---
    bins = np.linspace(-1.0, 1.0, 21)
    weights = np.ones_like(alpha_values) / len(alpha_values)
    ax1.hist(alpha_values, bins=bins, weights=weights, rwidth=0.85, 
             color=PROJECT_COLORS_HEX['PRIMARY'], label='Image Ratio')

    ax1.yaxis.set_major_formatter(PercentFormatter(1))
    ax1.set_xlabel("Per-Image K-α")
    ax1.set_ylabel("Ratio of Images", color=PROJECT_COLORS_HEX['PRIMARY'])
    ax1.tick_params(axis='y', labelcolor=PROJECT_COLORS_HEX['PRIMARY'])
    ax1.tick_params(axis='x', pad=5 * scale)
    ax1.grid(axis='y', alpha=0.75, linestyle='--', linewidth=0.5 * scale)
    ax1.set_xlim(-1.01, 1.01)

    ax1.axvline(mean_alpha, color=PROJECT_COLORS_HEX['SECONDARY_C'], linestyle='solid', linewidth=2 * scale, label=f'Mean: {mean_alpha:.2f}')
    ax1.axvline(global_alpha, color=PROJECT_COLORS_HEX['ACCENT'], linestyle='solid', linewidth=2 * scale, label=f'Global: {global_alpha:.2f}')

    # --- Secondary Y-Axis (Log Scale) ---
    ax2 = ax1.twinx()
    log_color = PROJECT_COLORS_HEX['PRIMARY_C']
    ax2.set_ylabel("Image Count (Log Scale)", color=log_color)
    ax2.tick_params(axis='y', labelcolor=log_color)
    ax2.set_yscale('log')

    counts, bin_edges = np.histogram(alpha_values, bins=bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    valid_indices = counts > 0
    if np.any(valid_indices):
        ax2.plot(bin_centers[valid_indices], counts[valid_indices], color=log_color, marker='o', linestyle='-', label='Image Count (Log)', linewidth=2 * scale, markersize=6 * scale)

    if np.any(counts > 0):
        min_val = np.min(counts[counts > 0])
        max_val = np.max(counts)
        ax2.set_ylim(min_val * 0.5, max_val * 2)
    else:
        ax2.set_ylim(0.1, 10)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    handles = [h1[0]] + h2 + h1[1:]
    labels = [l1[0]] + l2 + l1[1:]
    ax1.legend(handles, labels, loc='upper left')

    fig.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300 if file_format == 'png' else None, bbox_inches='tight')
    else:
        plt.show()
