"""
The main entry point and command router for the KaLOS CLI.
Uses jsonargparse to bind structured configuration dataclasses to their
respective decoupled execution pipelines, maintaining clean architectural boundaries.
"""

from jsonargparse import CLI

from kalos.iaa.kalos_execution import run_kalos_pipeline
from kalos.iaa.plotting_execution import run_plotting_pipeline
from kalos.iaa.empirical_disagreement import calculate_empirical_disagreement
from kalos.iaa.principled_configuration import derive_principled_configuration

from kalos.utils.yolo_to_kalos_coco import yolo_to_kalos_coco_pipeline

def main():
    """
    The main entry point for the Kalos CLI.
    """
    CLI(
        components={
            "calc-disagreement": calculate_empirical_disagreement,
            "configure": derive_principled_configuration,
            "execute": run_kalos_pipeline,
            "plot": run_plotting_pipeline,
            "convert-yolo": yolo_to_kalos_coco_pipeline, ## add YOLO to kalos-COCO conversion as a CLI command
        },
        description="Kalos: Inter-Annotator Agreement Evaluation Toolkit"
    )

if __name__ == "__main__":
    main()
