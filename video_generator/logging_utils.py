"""
Logging utilities for ProtoVideo.
Centralizes logger setup for all modules.
"""
import logging
import sys

def get_logger():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', stream=sys.stdout, force=True)
    return logging.getLogger(__name__) 