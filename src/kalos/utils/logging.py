"""
Logging utilities for the KaLOS toolkit.
Provides a centralized, tqdm-aware logging configuration to ensure that
CLI output and progress bars play nicely together without breaking the terminal.
"""

import logging
import sys
from tqdm import tqdm

class TqdmLoggingHandler(logging.Handler):
    """
    A logging handler that uses tqdm.write() to print log messages.
    This prevents log messages from breaking tqdm progress bars.
    """
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)

def setup_kalos_logging(level: str = "INFO"):
    """
    Configures the 'kalos' root logger for CLI usage.
    
    Args:
        level: The logging level to set (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """
    # Get the 'kalos' root logger
    logger = logging.getLogger("kalos")
    
    # Avoid duplicate handlers if setup is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create tqdm-aware handler
    handler = TqdmLoggingHandler()
    
    # Professional formatting: [LEVEL] - [Module Name] - Message
    formatter = logging.Formatter(
        fmt="%(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    # Prevent propagation to the root logger to avoid duplicate messages in CLI
    logger.propagate = False
