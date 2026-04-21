"""
Centralized theme and aesthetics manager for KaLOS visualizations.
Implements a Singleton pattern to control Matplotlib's global state and
provides declarative access to strictly defined, colorblind-friendly palettes.
"""

import logging
import warnings
import matplotlib.pyplot as plt
from matplotlib import colormaps as cmaps
from matplotlib.colors import ListedColormap
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# The 8 categories that define the Kalos visual identity
CATEGORIES = ['PRIMARY', 'SECONDARY', 'SUPPORT', 'PRIMARY_C', 'SECONDARY_C', 'SUPPORT_C', 'ACCENT', 'OKAY_ACCENT']
# The 6 categories used for the traffic light / agreement scales
TRAFFIC_LIGHT_ORDER = ['ACCENT', 'SECONDARY_C', 'PRIMARY_C', 'PRIMARY', 'SECONDARY', 'OKAY_ACCENT']

# Visually verified defaults
THEME_PRESETS = {
    "paper": ['#336B87', '#4F85A0', '#3E464B', '#B5854A', '#F59927', '#75644D', '#763626', '#A9FC6D'],
    "grayscale": ['#333333', '#555555', '#777777', '#999999', '#BBBBBB', '#DDDDDD', '#000000', '#FFFFFF'],
    "colorblind": ['#0072B2', '#56B4E9', '#000000', '#E69F00', '#F0E442', '#CC79A7', '#D55E00', '#009E73']
}

class ThemeManager:
    """Handles global plotting aesthetics and color palettes."""
    
    def __init__(self):
        self.colors: Dict[str, str] = {}
        self.font_scale: float = 1.0
        # Initialize with 'paper' defaults
        self.apply("paper")

    def apply(self, theme_name: str = "paper", font_family: str = "serif", font_name: Optional[str] = None, font_scale: float = 1.0, overrides: Optional[Dict[str, str]] = None, font_style: Optional[str] = 'normal'):
        """Sets the global style and registers colormaps."""
        self.font_scale = font_scale
        
        # 1. Resolve Colors
        base_list = THEME_PRESETS.get(theme_name, THEME_PRESETS["paper"])
        self.colors = {cat: color for cat, color in zip(CATEGORIES, base_list)}
        if overrides:
            self.colors.update(overrides)

        # 2. Configure Matplotlib Global state
        plt.rcParams.update({
            'font.family': font_family,
            'font.size': 12 * font_scale,
            'font.style': font_style,
            'pdf.fonttype': 42, 
            'ps.fonttype': 42
        })
        
        if font_name:
            family_key = f'font.{font_family}'
            if family_key in plt.rcParams:
                plt.rcParams[family_key] = [font_name] + list(plt.rcParams[family_key])
        elif font_family == "serif":
            plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']

        # 3. Register Colormaps
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            
            # Kalos Main Colormap (8 colors)
            kalos_cmap = ListedColormap([self.colors[c] for c in CATEGORIES], name='kalos')
            cmaps.register(cmap=kalos_cmap, force=True)
            cmaps.register(cmap=kalos_cmap.reversed(), force=True)

            # Traffic Light Agreement Colormap (6 colors)
            tl_cmap = ListedColormap([self.colors[c] for c in TRAFFIC_LIGHT_ORDER], name='TrafficLight')
            cmaps.register(cmap=tl_cmap, force=True)
            cmaps.register(cmap=tl_cmap.reversed(), force=True)

        logger.debug(f"Applied theme: {theme_name} (Scale: {font_scale})")

# --- Global Singleton and Proxy ---
theme_manager = ThemeManager()

class ThemeColorProxy(dict):
    """Minimal wrapper to access the active theme's color dictionary."""
    def __getitem__(self, key):
        return theme_manager.colors[key]
    def get(self, key, default=None):
        return theme_manager.colors.get(key, default)
    def items(self):
        return theme_manager.colors.items()

# This is the primary way modules access theme-aware colors
PROJECT_COLORS_HEX = ThemeColorProxy()
