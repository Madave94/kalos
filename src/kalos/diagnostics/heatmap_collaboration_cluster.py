import matplotlib.pyplot as plt
import numpy as np

def plot_collaboration_heatmap(collaboration_matrix, all_raters, output_file=None, file_format='png', label="Pairwise Image K-α"):
    """
    Plots a heatmap of the collaboration cluster analysis.
    """
    num_raters = len(all_raters)
    heatmap_data = np.zeros((num_raters, num_raters))

    for i, rater1 in enumerate(all_raters):
        for j, rater2 in enumerate(all_raters):
            if i == j:
                heatmap_data[i, j] = 1.0
            else:
                sorted_pair = tuple(sorted((rater1, rater2)))
                # Support both Mean (dict of dict of lists) and Global (dict of dict of floats)
                val = collaboration_matrix.get(sorted_pair[0], {}).get(sorted_pair[1], np.nan)
                
                if isinstance(val, list):
                    heatmap_data[i, j] = np.mean(val) if val else np.nan
                else:
                    heatmap_data[i, j] = val

    fig, ax = plt.subplots(figsize=(max(8, num_raters), max(6, num_raters * 0.8)))
    im = ax.imshow(heatmap_data, cmap='TrafficLight', vmin=0, vmax=1)

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(label, rotation=-90, va="bottom")

    ax.set_xticks(np.arange(num_raters))
    ax.set_yticks(np.arange(num_raters))
    ax.set_xticklabels(all_raters)
    ax.set_yticklabels(all_raters)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    threshold = im.norm(heatmap_data.max()) / 2.
    for i in range(num_raters):
        for j in range(num_raters):
            color = "w" if im.norm(heatmap_data[i, j]) < threshold else "black"
            ax.text(j, i, f"{heatmap_data[i, j]:.2f}",
                           ha="center", va="center", color=color)

    fig.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300 if file_format == 'png' else None, bbox_inches='tight')
    else:
        plt.show()
