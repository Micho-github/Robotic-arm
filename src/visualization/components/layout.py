import matplotlib.pyplot as plt
import matplotlib.patches as patches


def apply_layout(fig):
    """Apply shared layout: side panels, borders, and section headers."""
    # Panels
    fig.patches.extend([
        patches.Rectangle((0.00, 0.00), 0.24, 1.00, transform=fig.transFigure,
                          facecolor='#e6e6eb', alpha=0.95, zorder=0),
        patches.Rectangle((0.75, 0.00), 0.25, 1.00, transform=fig.transFigure,
                          facecolor='#e6e6eb', alpha=0.95, zorder=0),
    ])
    # Borders
    fig.lines.extend([
        plt.Line2D([0.00, 0.00], [0.00, 1.00], transform=fig.transFigure,
                   color='#c4c4cc', linewidth=1.0, alpha=0.9, zorder=1),
        plt.Line2D([0.24, 0.24], [0.00, 1.00], transform=fig.transFigure,
                   color='#c4c4cc', linewidth=1.0, alpha=0.9, zorder=1),
        plt.Line2D([0.75, 0.75], [0.00, 1.00], transform=fig.transFigure,
                   color='#c4c4cc', linewidth=1.0, alpha=0.9, zorder=1),
        plt.Line2D([1.00, 1.00], [0.00, 1.00], transform=fig.transFigure,
                   color='#c4c4cc', linewidth=1.0, alpha=0.9, zorder=1),
    ])
    # Headers
    fig.text(0.012, 0.32, "Legend", fontsize=12, fontweight='semibold', ha='left')
    fig.text(0.012, 0.74, "Status", fontsize=12, fontweight='semibold', ha='left')
    fig.text(0.76, 0.94, "Control Panel", fontsize=12, fontweight='semibold', ha='left')
    fig.text(0.76, 0.33, "Actions", fontsize=11, fontweight='semibold', ha='left', color='#333')
    fig.text(0.76, 0.74, "Values", fontsize=11, fontweight='semibold', ha='left', color='#333')


