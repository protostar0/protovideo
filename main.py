import os
import tempfile
import shutil
import uuid
import base64
import logging
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from moviepy import (
    ImageClip, VideoFileClip, AudioFileClip, concatenate_videoclips, CompositeVideoClip, TextClip, CompositeAudioClip
)
from gtts import gTTS
import requests
from moviepy import AudioClip

# --- Logging setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_DIR = tempfile.gettempdir()
OUTPUT_DIR = os.path.join(TEMP_DIR, "generated_videos")
os.makedirs(OUTPUT_DIR, exist_ok=True)

REEL_SIZE = (1080, 1920)  # width, height for 9:16 aspect ratio

# --- Models ---
class TextOverlay(BaseModel):
    content: str
    position: str = "center"  # center, top, bottom
    fontsize: int = 36
    color: str = "white"

class SceneInput(BaseModel):
    type: str  # 'image' or 'video'
    image: Optional[str] = None
    video: Optional[str] = None
    narration: Optional[str] = None
    narration_text: Optional[str] = None
    music: Optional[str] = None
    duration: int
    text: Optional[TextOverlay] = None

class VideoRequest(BaseModel):
    output_filename: str = Field(default_factory=lambda: f"output_{uuid.uuid4().hex}.mp4")
    scenes: List[SceneInput]

# --- Utility Functions ---
def download_asset(url_or_path: str) -> str:
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

def generate_narration(text: str) -> str:
    logger.info(f"Generating narration for text: {text[:60]}...")
    local_filename = os.path.join(TEMP_DIR, f"narration_{uuid.uuid4().hex}.mp3")
    try:
        tts = gTTS(text)
        tts.save(local_filename)
        logger.info(f"Generated narration at {local_filename}")
        return local_filename
    except Exception as e:
        logger.error(f"Failed to generate narration: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate narration: {e}")

def cleanup_files(filepaths: List[str]):
    logger.info(f"Cleaning up {len(filepaths)} temp files...")
    for path in filepaths:
        try:
            if os.path.exists(path):
                os.remove(path)
                logger.info(f"Deleted temp file: {path}")
        except Exception as e:
            logger.warning(f"Failed to delete temp file {path}: {e}")

def render_scene(scene: SceneInput) -> (str, List[str]):
    logger.info(f"Rendering scene: {scene}")
    temp_files = []
    video_clip = None
    audio_clips = []
    # Handle image or video
    if scene.type == "image":
        if not scene.image:
            logger.error("Image URL/path required for image scene.")
            raise HTTPException(status_code=400, detail="Image URL/path required for image scene.")
        image_path = download_asset(scene.image)
        temp_files.append(image_path)
        # Start with a slightly larger image to allow zoom-in
        base_scale = 5  # Start at 110% of REEL_SIZE so we can zoom in
        base_width = int(REEL_SIZE[0] * base_scale)
        base_height = int(REEL_SIZE[1] * base_scale)
        video_clip = ImageClip(image_path).with_duration(scene.duration)
        video_clip = video_clip.resized(new_size=(base_width, base_height))
        # Apply Ken Burns zoom-in effect (from 100% to 110%)
        video_clip = video_clip.resized(lambda t: 1.0 + 0.5 * (t / scene.duration))
        # Center in REEL_SIZE frame
        video_clip = video_clip.with_background_color(size=REEL_SIZE, color=(0,0,0), pos='center')
        logger.info(f"Created ImageClip for {image_path} with REEL_SIZE and zoom-in effect")
    elif scene.type == "video":
        if not scene.video:
            logger.error("Video URL/path required for video scene.")
            raise HTTPException(status_code=400, detail="Video URL/path required for video scene.")
        video_path = download_asset(scene.video)
        temp_files.append(video_path)
        video_clip = VideoFileClip(video_path)
        video_clip = video_clip.subclipped(0, min(scene.duration, video_clip.duration))
        if scene.duration < video_clip.duration:
            video_clip = video_clip.subclipped(0, scene.duration)
        else:
            video_clip = video_clip.with_duration(scene.duration)
        video_clip = video_clip.resized(height=REEL_SIZE[1])  # fit height
        video_clip = video_clip.with_background_color(size=REEL_SIZE, color=(0,0,0), pos='center')
        logger.info(f"Created VideoFileClip for {video_path} with REEL_SIZE")
    else:
        logger.error(f"Unknown scene type: {scene.type}")
        raise HTTPException(status_code=400, detail=f"Unknown scene type: {scene.type}")
    # Handle text overlay
    if scene.text:
        logger.info(f"Adding text overlay: {scene.text.content}")
        txt_clip = TextClip(
            font="Arial.ttf",
            text=scene.text.content,
            font_size=scene.text.fontsize,
            color=scene.text.color
        ).with_duration(scene.duration)
        if scene.text.position == "center":
            txt_clip = txt_clip.with_position('center')
        elif scene.text.position == "top":
            txt_clip = txt_clip.with_position(('center', 'top'))
        elif scene.text.position == "bottom":
            txt_clip = txt_clip.with_position(('center', 'bottom'))
        video_clip = CompositeVideoClip([video_clip, txt_clip])
        logger.info("Text overlay added.")
    # Handle narration
    narration_path = None
    if scene.narration:
        narration_path = download_asset(scene.narration)
        temp_files.append(narration_path)
        logger.info(f"Added narration from file: {narration_path}")
    elif scene.narration_text:
        narration_path = generate_narration(scene.narration_text)
        temp_files.append(narration_path)
        logger.info(f"Added narration from text: {narration_path}")
    if narration_path:
        narration_clip = AudioFileClip(narration_path)
        if narration_clip.duration < video_clip.duration:
            # Pad with silence
            silence = AudioClip(lambda t: 0, duration=video_clip.duration - narration_clip.duration)
            narration_padded = CompositeAudioClip([
                narration_clip,
                silence.with_start(narration_clip.duration)
            ])
            narration_padded = narration_padded.with_duration(video_clip.duration)
        else:
            narration_padded = narration_clip.subclipped(0, video_clip.duration)
        video_clip = video_clip.with_audio(narration_padded)
    # Handle music
    if scene.music:
        music_path = download_asset(scene.music)
        temp_files.append(music_path)
        audio_clips.append(AudioFileClip(music_path).with_volume_scaled(0.3).with_duration(scene.duration))
        logger.info(f"Added background music: {music_path}")
    # Mix audio
    if audio_clips:
        logger.info(f"Mixing {len(audio_clips)} audio tracks for scene.")
        composite_audio = CompositeAudioClip(audio_clips)
        video_clip = video_clip.with_audio(composite_audio)
    # Export scene
    scene_output = os.path.join(TEMP_DIR, f"scene_{uuid.uuid4().hex}.mp4")
    logger.info(f"Exporting scene to {scene_output}")
    video_clip.write_videofile(
        scene_output,
        fps=24,
        codec="libx264",
        audio_codec="aac",
        temp_audiofile=f"{scene_output}.temp_audio.m4a",
        remove_temp=True,
        logger=None
    )
    temp_files.append(scene_output)
    video_clip.close()
    logger.info(f"Scene rendered and saved: {scene_output}")
    return scene_output, temp_files

# --- API Endpoint ---
@app.post("/generate")
def generate_video(request: VideoRequest, background_tasks: BackgroundTasks):
    logger.info(f"Received video generation request: {request.output_filename}")
    temp_files = []
    scene_files = []
    try:
        for idx, scene in enumerate(request.scenes):
            logger.info(f"Processing scene {idx+1}/{len(request.scenes)}")
            scene_file, files_to_clean = render_scene(scene)
            scene_files.append(scene_file)
            temp_files.extend(files_to_clean)
        # Concatenate scenes
        logger.info(f"Concatenating {len(scene_files)} scenes...")
        clips = [VideoFileClip(f) for f in scene_files]
        final_clip = concatenate_videoclips(clips, method="compose")
        output_path = os.path.join(OUTPUT_DIR, f"{uuid.uuid4().hex}_{request.output_filename}")
        logger.info(f"Exporting final video to {output_path}")
        final_clip.write_videofile(
            output_path,
            fps=24,
            codec="libx264",
            audio_codec="aac",
            temp_audiofile=f"{output_path}.temp_audio.m4a",
            remove_temp=True,
            logger=None
        )
        final_clip.close()
        for c in clips:
            c.close()
        # Clean up temp files in background
        background_tasks.add_task(cleanup_files, temp_files)
        logger.info(f"Video generation complete: {output_path}")
        # Return download link
        return JSONResponse({"download_url": f"/download/{os.path.basename(output_path)}"})
    except Exception as e:
        logger.error(f"Video generation failed: {e}")
        cleanup_files(temp_files)
        raise HTTPException(status_code=500, detail=f"Video generation failed: {e}")

@app.get("/download/{filename}")
def download_video(filename: str):
    file_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(file_path):
        logger.error(f"File not found for download: {filename}")
        raise HTTPException(status_code=404, detail="File not found.")
    logger.info(f"Serving file for download: {filename}")
    return FileResponse(file_path, media_type="video/mp4", filename=filename)

@app.get("/")
def root():
    logger.info("Root endpoint accessed.")
    return {
        "message": "Welcome to the Video Generation API! Use POST /generate with your video project JSON to create a video.",
        "usage": {
            "endpoint": "/generate",
            "method": "POST",
            "json_structure": {
                "output_filename": "string (e.g. my_video.mp4)",
                "scenes": [
                    {
                        "type": "image or video",
                        "image": "url or path (for image scenes)",
                        "video": "url or path (for video scenes)",
                        "narration": "url or path to mp3 (optional)",
                        "narration_text": "string (optional, will use gTTS)",
                        "music": "url or path to mp3 (optional)",
                        "duration": "int (seconds)",
                        "text": {
                            "content": "string",
                            "position": "center|top|bottom",
                            "fontsize": "int",
                            "color": "string"
                        }
                    }
                ]
            }
        }
    } 