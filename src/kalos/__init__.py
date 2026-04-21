import logging

# Set up the 'kalos' root logger with a NullHandler.
# This follows library best practices by preventing "No handler found" 
# warnings when kalos is imported as an API.
logging.getLogger(__name__).addHandler(logging.NullHandler())

from .iaa import (
    calculate_iaa,
    vision_alpha,
    calculate_empirical_disagreement,
    derive_principled_configuration
)
from .iaa.kalos_execution import run_kalos_pipeline
from .iaa.plotting_execution import run_plotting_pipeline
from .utils.yolo_to_kalos_coco import yolo_to_kalos_coco_pipeline

__all__ = [
    "calculate_iaa",
    "vision_alpha",
    "calculate_empirical_disagreement",
    "derive_principled_configuration",
    "run_kalos_pipeline",
    "run_plotting_pipeline",
    "yolo_to_kalos_coco_pipeline"
]
