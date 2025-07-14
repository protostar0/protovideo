"""
Image utilities for ProtoVideo.
Handles downloading images and generating images from prompts.
"""
from typing import Optional
import os
import uuid
import requests
import logging
from fastapi import HTTPException
from .generate_image import generate_image_from_prompt as _generate_image_from_prompt
from .config import Config

TEMP_DIR = Config.TEMP_DIR

def download_asset(url_or_path: str) -> str:
    """
    Download an asset from a URL or return the local path if it exists.
    Returns the local file path.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Downloading asset: {url_or_path}")
    if url_or_path.startswith("http://") or url_or_path.startswith("https://"):
        local_filename = os.path.join(TEMP_DIR, f"asset_{uuid.uuid4().hex}{os.path.splitext(url_or_path)[-1]}")
        try:
            r = requests.get(url_or_path, stream=True, timeout=60)
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info(f"Downloaded asset to {local_filename}")
            return local_filename
        except Exception as e:
            logger.error(f"Failed to download asset: {url_or_path} ({e})")
            raise HTTPException(status_code=400, detail=f"Failed to download asset: {url_or_path} ({e})")
    elif os.path.exists(url_or_path):
        logger.info(f"Using local asset: {url_or_path}")
        return url_or_path
    else:
        logger.error(f"Asset not found: {url_or_path}")
        raise HTTPException(status_code=400, detail=f"Asset not found: {url_or_path}")

def generate_image_from_prompt(prompt: str, api_key: str, out_path: str) -> str:
    """
    Generate an image from a text prompt using OpenAI API and save to out_path.
    Returns the path to the generated image.
    """
    return _generate_image_from_prompt(prompt, api_key, out_path) 