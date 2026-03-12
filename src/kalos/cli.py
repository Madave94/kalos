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
from kalos.config import (
    EmpiricalDisagreementConfig, 
    PrincipledConfigurationConfig, 
    KaLOSProjectConfig
)

def calc_disagreement_wrapper(cfg: EmpiricalDisagreementConfig):
    """
    Wrapper to bridge CLI config to the decoupled empirical disagreement API.

    Args:
        cfg (EmpiricalDisagreementConfig): Configuration object containing 
            paths and parameters for generating D_o and D_e distributions.
    """
    return calculate_empirical_disagreement(
        annotation_file=str(cfg.annotation_file),
        output_file=str(cfg.output_file),
        similarity_func=cfg.similarity_func,
        only_with_annotations=cfg.only_with_annotations,
        log_level=cfg.log_level
    )

def configure_wrapper(cfg: PrincipledConfigurationConfig):
    """
    Wrapper to bridge CLI config to the decoupled principled configuration API.

    Args:
        cfg (PrincipledConfigurationConfig): Configuration object containing 
            disagreement file paths and global plotting settings.
    """
    return derive_principled_configuration(
        disagreement_files=[str(f) for f in cfg.disagreement_files],
        output_dir=str(cfg.output_dir),
        plotting=cfg.plotting,
        log_level=cfg.log_level
    )

def main():
    """
    The main entry point for the Kalos CLI.
    """
    CLI(
        components={
            "calc-disagreement": calc_disagreement_wrapper,
            "configure": configure_wrapper,
            "execute": run_kalos_pipeline,
            "plot": run_plotting_pipeline,
        },
        description="Kalos: Inter-Annotator Agreement Evaluation Toolkit"
    )

if __name__ == "__main__":
    main()
