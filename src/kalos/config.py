"""
Defines the configuration data models for the KaLOS toolkit.
Uses dataclasses to provide strict typing and default values for all CLI commands,
ensuring clean separation between user input and internal execution logic.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Literal, Dict
from jsonargparse.typing import Path_fr, Path_dw, Path_fc

@dataclass
class PlotSettings:
    """Granular settings for a specific diagnostic plot."""
    enabled: bool = False
    output_path: Optional[Path_dw] = None
    font_scale: Optional[float] = None
    font_family: Optional[str] = None
    theme: Optional[str] = None
    color_overrides: Optional[Dict[str, str]] = None

@dataclass
class PlottingConfig:
    """Global plotting configuration with per-plot overrides."""
    plot_all: bool = False
    output_folder: Optional[Path_dw] = None
    plot_format: Literal["png", "pdf"] = "png"
    
    # Global Defaults
    font_scale: float = 1.0
    font_family: str = "serif"
    theme: str = "paper"
    color_overrides: Optional[Dict[str, str]] = None

    # Granular Plot Settings
    alpha_distribution: PlotSettings = field(default_factory=PlotSettings)
    collaboration_heatmap: PlotSettings = field(default_factory=PlotSettings)
    localization_sensitivity: PlotSettings = field(default_factory=PlotSettings)
    annotator_vitality: PlotSettings = field(default_factory=PlotSettings)
    class_difficulty: PlotSettings = field(default_factory=PlotSettings)


@dataclass
class KaLOSProjectConfig:
    """Master configuration for the KaLOS agreement pipeline (Math + Visuals)."""
    annotation_file: Path_fr
    task: Literal['bbox', 'segm', '3D_VIS', 'keypoints']
    method: str
    threshold_func: str
    cost_func: str
    similarity_threshold: float
    annotation_type: Literal['coco-json', 'lidc-idri-json'] = "coco-json"
    only_with_instances: bool = False

    # Downstream Analysis Flags
    calculate_vitality: bool = False
    calculate_difficulty: bool = False
    calculate_intra_iaa: bool = False
    collaboration_clusters: bool = False
    localization_sensitivity_thresholds: Optional[List[float]] = None

    # Results Export
    output_results: Optional[Path_dw] = None

    # Nested Plotting Group (Shared by execute and plot)
    plotting: PlottingConfig = field(default_factory=PlottingConfig)

    # System Flags
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"


@dataclass
class EmpiricalDisagreementConfig:
    """Configuration for calculating observed and expected disagreement."""
    annotation_file: Path_fr
    output_file: Path_fc
    similarity_func: str
    only_with_annotations: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"


@dataclass
class PrincipledConfigurationConfig:
    """Configuration for deriving principled configuration."""
    disagreement_files: List[Path_fr]
    output_dir: Path_dw
    plotting: PlottingConfig = field(default_factory=PlottingConfig)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
