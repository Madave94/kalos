import matplotlib.pyplot as plt
import numpy as np
import logging
from kalos.utils.theme_manager import theme_manager, PROJECT_COLORS_HEX

logger = logging.getLogger(__name__)

def plot_annotator_vitality(annotator_vitalities, output_file=None, file_format='png'):
    """
    Plots the classic influence bubble plot for Mean Annotator Vitality.
    """
    scale = theme_manager.font_scale
    
    raters = sorted(annotator_vitalities.keys())

    mean_vits = [np.mean(annotator_vitalities[r]) for r in raters]
    counts = [len(annotator_vitalities[r]) for r in raters]
    
    # Normalize bubble sizes
    max_count = max(counts) if counts else 1
    sizes = [(c / max_count) * 1000 * scale for c in counts]

    fig, ax = plt.subplots(figsize=(max(8, len(raters)), 6))
    
    # Plot scatter
    ax.scatter(raters, mean_vits, s=sizes, 
               color=PROJECT_COLORS_HEX['PRIMARY'], 
               alpha=0.7, edgecolors=PROJECT_COLORS_HEX['SUPPORT'])

    # Add counts as text inside bubbles
    for i, count in enumerate(counts):
        ax.annotate(str(count), (raters[i], mean_vits[i]), 
                    ha='center', va='center', 
                    color=PROJECT_COLORS_HEX["SUPPORT_C"],
                    fontsize=10 * scale, fontweight='bold')

    ax.set_xlabel("Annotator ID")
    ax.set_ylabel("Mean K-α Vitality")
    
    # Add a horizontal line at 0
    ax.axhline(0, color=PROJECT_COLORS_HEX['SUPPORT_C'], linestyle='--', linewidth=1 * scale)

    ax.grid(True, linestyle=':', alpha=0.6)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    if mean_vits:
        y_min, y_max = min(mean_vits), max(mean_vits)
        y_range = y_max - y_min if y_max != y_min else 0.1
        ax.set_ylim(y_min - y_range * 0.2, y_max + y_range * 0.2)

    fig.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300 if file_format == 'png' else None, bbox_inches='tight')
    else:
        plt.show()
