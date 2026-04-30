"""
plot_style.py — Shared plot style for all course notebooks.

Usage
-----
Import and call setup_plot_style() at the top of every notebook setup cell:

    from plot_style import setup_plot_style, COLORBLIND_PALETTE
    setup_plot_style()

The colorblind-safe palette is derived from seaborn's "colorblind" preset
(Wong 2011) and is safe for the most common types of color vision deficiency
(deuteranopia, protanopia, tritanopia).
"""

import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Colorblind-safe palette (seaborn "colorblind", 8 colors, Wong 2011)
# ---------------------------------------------------------------------------

COLORBLIND_PALETTE = sns.color_palette("colorblind").as_hex()
# ['#0173b2', '#de8f05', '#029e73', '#d55e00',
#  '#cc78bc', '#ca9161', '#fbafe4', '#949494']


# ---------------------------------------------------------------------------
# Style setup
# ---------------------------------------------------------------------------

def setup_plot_style() -> None:
    """Apply the standard course plot style to all subsequent figures.

    Sets:
    - Grid style: seaborn-v0_8-whitegrid
    - Color palette: colorblind-safe (8 colors)
    - DPI: 120
    - Font size: 11 pt (DejaVu Sans — supports Unicode / Ukrainian glyphs)
    - Spines: top and right removed
    - Default figure size: 8 × 5 inches
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_palette("colorblind")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "figure.figsize": (8, 5),
            "font.size": 11,
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
