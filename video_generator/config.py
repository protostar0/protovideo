"""
Configuration for ProtoVideo.
Centralizes all environment-based and default settings.
"""
import os
from pathlib import Path

class Config:
    # OpenAI API
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
    # Cloudflare R2
    R2_ENDPOINT_URL = os.environ.get("R2_ENDPOINT_URL", "")
    R2_ACCESS_KEY_ID = os.environ.get("R2_ACCESS_KEY_ID", "")
    R2_SECRET_ACCESS_KEY = os.environ.get("R2_SECRET_ACCESS_KEY", "")
    R2_PUBLIC_BASE_URL = os.environ.get("R2_PUBLIC_BASE_URL", "")
    R2_BUCKET_NAME = os.environ.get("R2_BUCKET_NAME", "")
    # Temp and output directories
    TEMP_DIR = os.getenv('TMPDIR', '/tmp')
    OUTPUT_DIR = os.environ.get("OUTPUT_DIR", str(Path(TEMP_DIR) / "generated_videos"))
    # Video settings
    REEL_SIZE = (1080, 1920)
    FPS = 24
    # API Key for your service
    API_KEY = os.environ.get("PROTOVIDEO_API_KEY", "N8S6R_TydmHr58LoUzYZf9v2gRkcfWZemz1zWZ5WMkE") 