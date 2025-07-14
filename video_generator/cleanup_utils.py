"""
Cleanup utilities for ProtoVideo.
Handles deletion of temporary files and memory cleanup.
"""
from typing import List
import os
import logging

def cleanup_files(filepaths: List[str]) -> None:
    """
    Delete a list of files from disk if they exist.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Cleaning up {len(filepaths)} temp files...")
    for path in filepaths:
        try:
            if os.path.exists(path):
                os.remove(path)
                logger.info(f"Deleted temp file: {path}")
        except Exception as e:
            logger.warning(f"Failed to delete temp file {path}: {e}") 