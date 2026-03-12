import matplotlib.pyplot as plt
import numpy as np
import logging
from kalos.utils.theme_manager import theme_manager, PROJECT_COLORS_HEX

logger = logging.getLogger(__name__)

def plot_class_difficulty(mean_difficulties, global_difficulties, output_file=None, file_format='png'):
    """
    Plots a grouped horizontal bar chart for class difficulty (Mean vs Global Alpha).
    """
    scale = theme_manager.font_scale
    
    # 1. Prepare and align data
    classes = []
    m_alphas = []
    g_alphas = []
    
    for cls_name, data in mean_difficulties.items():
        if data['alphas']:
            classes.append(cls_name)
            m_alphas.append(np.mean(data['alphas']))
            # Get corresponding global alpha
            g_alpha = global_difficulties.get(cls_name, {}).get("alpha", 0.0)
            g_alphas.append(g_alpha)

    # 2. Sort by Mean Alpha (descending)
    sorted_indices = np.argsort(m_alphas)[::-1]
    classes = [classes[i] for i in sorted_indices]
    m_alphas = [m_alphas[i] for i in sorted_indices]
    g_alphas = [g_alphas[i] for i in sorted_indices]

    # 3. Plotting
    fig, ax = plt.subplots(figsize=(10, max(6, len(classes) * 0.8)))
    
    y_pos = np.arange(len(classes))
    height = 0.35  # height of the bars

    # Mean Alpha Bars (Primary)
    rects1 = ax.barh(y_pos - height/2, m_alphas, height, 
                     label='Mean Image K-α', color=PROJECT_COLORS_HEX['PRIMARY'])
    
    # Global Alpha Bars (Accent)
    rects2 = ax.barh(y_pos + height/2, g_alphas, height, 
                     label='Global Dataset K-α', color=PROJECT_COLORS_HEX['ACCENT'])

    # Labels and Aesthetics
    ax.set_yticks(y_pos)
    ax.set_yticklabels(classes)
    ax.invert_yaxis()
    ax.set_xlabel("K-α")
    ax.legend(loc='lower right', fontsize=10 * scale)
    
    # Grid and limits
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    all_vals = m_alphas + g_alphas
    max_val = max(all_vals) if all_vals else 1.0
    ax.set_xlim(min(0, min(all_vals)), max(1.05, max_val + 0.1))

    # Add values text
    for i, (m, g) in enumerate(zip(m_alphas, g_alphas)):
        ax.text(m + 0.01, i - height/2, f"{m:.2f}", va='center', fontsize=9 * scale)
        ax.text(g + 0.01, i + height/2, f"{g:.2f}", va='center', fontsize=9 * scale)

    fig.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300 if file_format == 'png' else None, bbox_inches='tight')
    else:
        plt.show()
