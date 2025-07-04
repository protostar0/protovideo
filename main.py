import os
import tempfile
import shutil
import uuid
import base64
import logging
import time
import threading
import whisper
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Header
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from moviepy import (
    ImageClip, VideoFileClip, AudioFileClip, concatenate_videoclips, CompositeVideoClip, TextClip, CompositeAudioClip
)
from gtts import gTTS
import requests
from moviepy import AudioClip
from moviepy import concatenate_videoclips
from moviepy.video.fx import CrossFadeIn, CrossFadeOut
import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
import warnings
warnings.filterwarnings("ignore")
from transformers import logging as loggingts
loggingts.set_verbosity_error()
# Detect device (Mac with M1/M2/M3/M4)
device = "mps" if torch.backends.mps.is_available() else "cpu"
map_location = torch.device(device)
import os
os.environ['TRANSFORMERS_NO_PROGRESS_BAR'] = '1'
torch_load_original = torch.load
def patched_torch_load(*args, **kwargs):
    if 'map_location' not in kwargs:
        kwargs['map_location'] = map_location
    return torch_load_original(*args, **kwargs)

torch.load = patched_torch_load

model = ChatterboxTTS.from_pretrained(device=device)

# --- Logging setup ---
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

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
    narration_text: Optional[str] = None  # <-- new field for global narration

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

def generate_narration2(text: str) -> str:
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


def generate_narration(text: str) -> str:
    logger.info(f"Generating narration for text: {text[:60]}...")
    local_filename = os.path.join(TEMP_DIR, f"narration_{uuid.uuid4().hex}.mp3")
    try:
        wav = model.generate(
        text, 
        # audio_prompt_path=AUDIO_PROMPT_PATH,
        exaggeration=0.5,
        cfg_weight=0.5
        )
        ta.save(local_filename, wav, model.sr)
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

from moviepy.video.fx import MultiplyColor

def render_scene(scene: SceneInput, use_global_narration=False) -> (str, List[str]):
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
        video_clip = ImageClip(image_path).with_duration(scene.duration)
        # Resize to fit REEL_SIZE, preserving aspect ratio (no cropping)
        video_clip = video_clip.resized(height=REEL_SIZE[1])
        if video_clip.w > REEL_SIZE[0]:
            video_clip = video_clip.resized(width=REEL_SIZE[0])
        # Apply Ken Burns (zoom-in) effect to the image only (before padding)
        def zoom(t):
            return 1.0 + 0.5 * (t / scene.duration)
        video_clip = video_clip.resized(zoom)
        # Pad to REEL_SIZE (after zoom)
        video_clip = video_clip.with_background_color(size=REEL_SIZE, color=(0,0,0), pos='center')
        logger.info(f"Created ImageClip for {image_path} with REEL_SIZE, full image visible, and correct zoom-in effect")
        # Darken the image for better text readability
        video_clip = video_clip.with_effects([MultiplyColor(0.5)])
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
        video_clip = video_clip.resized(height=REEL_SIZE[1])
        video_clip = video_clip.with_background_color(size=REEL_SIZE, color=(0,0,0), pos='center')
        logger.info(f"Created VideoFileClip for {video_path} with REEL_SIZE")
        # Darken the video for better text readability
        video_clip = video_clip.with_effects([MultiplyColor(0.5)])
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
    # Handle narration (skip if using global narration)
    narration_path = None
    if not use_global_narration:
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
    start_time = time.time()
    temp_files = []
    scene_files = []
    try:
        use_global_narration = bool(request.narration_text)
        narration_path = None
        narration_duration = None
        # If global narration, generate audio first and determine scene durations
        if use_global_narration:
            narration_path = generate_narration(request.narration_text)
            temp_files.append(narration_path)
            narration_clip = AudioFileClip(narration_path)
            narration_duration = narration_clip.duration
            num_scenes = len(request.scenes)
            if num_scenes > 0:
                per_scene_duration = narration_duration / num_scenes
                for scene in request.scenes:
                    scene.duration = int(round(per_scene_duration))
        for idx, scene in enumerate(request.scenes):
            logger.info(f"Processing scene {idx+1}/{len(request.scenes)}")
            scene_file, files_to_clean = render_scene(scene, use_global_narration=use_global_narration)
            scene_files.append(scene_file)
            temp_files.extend(files_to_clean)
        logger.info(f"Concatenating {len(scene_files)} scenes...")
        clips = [VideoFileClip(f) for f in scene_files]
        CROSSFADE_DURATION = 1.0
        for i in range(len(clips)):
            if i > 0:
                clips[i] = clips[i].with_effects([CrossFadeIn(CROSSFADE_DURATION)])
            if i < len(clips) - 1:
                clips[i] = clips[i].with_effects([CrossFadeOut(CROSSFADE_DURATION)])
        final_clip = concatenate_videoclips(clips, method="compose", padding=-CROSSFADE_DURATION)

        # Ensure video is at least as long as the narration audio
        if use_global_narration and narration_clip and final_clip.duration < narration_clip.duration:
            final_clip = final_clip.with_duration(narration_clip.duration + 0.2)  # Add a small buffer

        output_path = os.path.join(OUTPUT_DIR, f"{uuid.uuid4().hex}_{request.output_filename}")

        # If global narration_text is provided, overlay it on the final video
        if use_global_narration:
            if narration_clip.duration < final_clip.duration:
                silence = AudioClip(lambda t: 0, duration=final_clip.duration - narration_clip.duration)
                narration_padded = CompositeAudioClip([
                    narration_clip,
                    silence.with_start(narration_clip.duration)
                ]).with_duration(final_clip.duration)
            else:
                narration_padded = narration_clip.subclipped(0, final_clip.duration)
            final_clip = final_clip.with_audio(narration_padded)

            # --- Whisper-based animated phrase subtitles ---
            try:
                subtitle_clips = generate_whisper_phrase_subtitles(narration_path, final_clip, words_per_line=4, font_size=60)
                # subtitle_clips = generate_whisper_phrase_subtitles(narration_path, final_clip, font_size=60)
                final_clip = CompositeVideoClip([final_clip] + subtitle_clips)
            except Exception as e:
                logger.warning(f"Whisper subtitle generation failed: {e}")
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
        background_tasks.add_task(cleanup_files, temp_files)
        elapsed = time.time() - start_time
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)
        logger.info(f"Video generation complete: {output_path} (Time taken: {mins}m {secs}s)")
        return JSONResponse({"download_url": f"/download/{os.path.basename(output_path)}", "generation_time": f"{mins}m {secs}s"})
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

# --- Whisper-based phrase subtitle function ---
# def generate_whisper_phrase_subtitles(audio_path, video_clip, words_per_line=4, font_size=60):
#     import whisper
#     model = whisper.load_model("base")
#     result = model.transcribe(audio_path, word_timestamps=True)
#     words = []
#     for segment in result['segments']:
#         for word in segment.get('words', []):
#             words.append(word)
#     # Group words into lines
#     lines = []
#     for i in range(0, len(words), words_per_line):
#         chunk = words[i:i+words_per_line]
#         if not chunk:
#             continue
#         line_text = ' '.join([w['word'] for w in chunk])
#         start = chunk[0]['start']
#         end = chunk[-1]['end']
#         lines.append({'text': line_text, 'start': start, 'end': end})
#     # Create subtitle clips
#     subtitle_clips = []
#     for line in lines:
#         txt_clip = (
#             TextClip(
#                 text=line['text'].upper(),
#                 font="Arial",
#                 font_size=font_size,
#                 color="white",
#                 stroke_color="black",
#                 stroke_width=3,
#                 method="label",
#                 size=(video_clip.w - 120, None)
#             )
#             .with_position(("center", int(video_clip.h * 0.5)))
#             .with_start(line['start'])
#             .with_duration(line['end'] - line['start'])
#             .with_opacity(0.95)
#         )
#         subtitle_clips.append(txt_clip)
#     return subtitle_clips 

def generate_whisper_phrase_subtitles(audio_path, video_clip, words_per_line=4, font_size=50):
    model = whisper.load_model("base")

    result = model.transcribe(audio_path, word_timestamps=True)

    all_words = []
    for segment in result['segments']:
        all_words.extend(segment.get('words', []))

    # Group words into lines
    lines = []
    for i in range(0, len(all_words), words_per_line):
        chunk = all_words[i:i + words_per_line]
        if not chunk:
            continue
        line_text = ' '.join([w['word'].strip() for w in chunk])
        start = chunk[0]['start']
        end = chunk[-1]['end']
        lines.append({'text': line_text.upper(), 'start': start, 'end': end, 'words': chunk})

    # Create subtitle clips
    subtitle_clips = []

    for line in lines:
        # Base line in white
        base_clip = (
            TextClip(
                text = line['text']+"\n_",
                font="./montserrat/Montserrat-Black.ttf",
                font_size=font_size,
                color="white",
                stroke_color="black",
                stroke_width=4,
                method="caption",
                text_align="center",
                size=(video_clip.w - 120, None)
            )
            .with_position(("center", int(video_clip.h * 0.4)))
            .with_start(line['start'])
            .with_duration(line['end'] - line['start'])
        )
        subtitle_clips.append(base_clip)

    return subtitle_clips

# In-memory task store
tasks = {}  # task_id: {status, result, error}

def video_generation_worker(task_id, request_dict):
    try:
        tasks[task_id]["status"] = "inprogress"
        # Reuse the generate_video logic, but don't use FastAPI request/response
        # Simulate a request object
        class DummyRequest:
            pass
        dummy_request = DummyRequest()
        for k, v in request_dict.items():
            setattr(dummy_request, k, v)
        # Call the main video generation logic
        # We need to call the core logic, not the FastAPI endpoint
        # So, refactor the main logic into a helper function
        result = generate_video_core(request_dict)
        tasks[task_id]["status"] = "finished"
        tasks[task_id]["result"] = result
    except Exception as e:
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"] = str(e)

# Refactor the main video generation logic into a helper function
def generate_video_core(request_dict):
    # This is a copy of the main logic from generate_video, but using dict input/output
    # (You can refactor your code to avoid duplication if you wish)
    import copy
    request = copy.deepcopy(request_dict)
    temp_files = []
    scene_files = []
    use_global_narration = bool(request.get("narration_text"))
    narration_path = None
    narration_duration = None
    try:
        if use_global_narration:
            narration_path = generate_narration(request["narration_text"])
            temp_files.append(narration_path)
            narration_clip = AudioFileClip(narration_path)
            narration_duration = narration_clip.duration
            num_scenes = len(request["scenes"])
            if num_scenes > 0:
                per_scene_duration = narration_duration / num_scenes
                for scene in request["scenes"]:
                    scene["duration"] = int(round(per_scene_duration))
        for idx, scene in enumerate(request["scenes"]):
            scene_file, files_to_clean = render_scene(SceneInput(**scene), use_global_narration=use_global_narration)
            scene_files.append(scene_file)
            temp_files.extend(files_to_clean)
        clips = [VideoFileClip(f) for f in scene_files]
        CROSSFADE_DURATION = 1.0
        for i in range(len(clips)):
            if i > 0:
                clips[i] = clips[i].with_effects([CrossFadeIn(CROSSFADE_DURATION)])
            if i < len(clips) - 1:
                clips[i] = clips[i].with_effects([CrossFadeOut(CROSSFADE_DURATION)])
        final_clip = concatenate_videoclips(clips, method="compose", padding=-CROSSFADE_DURATION)
        if use_global_narration and narration_clip and final_clip.duration < narration_clip.duration:
            final_clip = final_clip.with_duration(narration_clip.duration + 0.2)
        output_path = os.path.join(OUTPUT_DIR, f"{uuid.uuid4().hex}_{request['output_filename']}")
        if use_global_narration:
            if narration_clip.duration < final_clip.duration:
                silence = AudioClip(lambda t: 0, duration=final_clip.duration - narration_clip.duration)
                narration_padded = CompositeAudioClip([
                    narration_clip,
                    silence.with_start(narration_clip.duration)
                ]).with_duration(final_clip.duration)
            else:
                narration_padded = narration_clip.subclipped(0, final_clip.duration)
            final_clip = final_clip.with_audio(narration_padded)
            try:
                subtitle_clips = generate_whisper_phrase_subtitles(narration_path, final_clip, words_per_line=4, font_size=60)
                final_clip = CompositeVideoClip([final_clip] + subtitle_clips)
            except Exception as e:
                logger.warning(f"Whisper subtitle generation failed: {e}")
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
        cleanup_files(temp_files)
        return {"download_url": f"/download/{os.path.basename(output_path)}"}
    except Exception as e:
        cleanup_files(temp_files)
        raise

API_KEY = "N8S6R_TydmHr58LoUzYZf9v2gRkcfWZemz1zWZ5WMkE"  # Change this to your actual secret key

def require_api_key(request: Request):
    api_key = request.headers.get("x-api-key")
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")

# Wrap endpoints with API key check
from functools import wraps
import inspect

def api_key_required(endpoint):
    @wraps(endpoint)
    async def wrapper(*args, **kwargs):
        request = kwargs.get('request')
        if not request:
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
        if not request:
            raise HTTPException(status_code=401, detail="Missing request context for API key check.")
        require_api_key(request)
        if inspect.iscoroutinefunction(endpoint):
            return await endpoint(*args, **kwargs)
        else:
            return endpoint(*args, **kwargs)
    return wrapper

# Update endpoints to require API key
@app.post("/generate-task")
@api_key_required
def generate_task(request: Request, video_request: VideoRequest):
    task_id = str(uuid.uuid4())
    tasks[task_id] = {"status": "queued"}
    thread = threading.Thread(target=video_generation_worker, args=(task_id, video_request.dict()))
    thread.start()
    return {"task_id": task_id, "status": "queued"}

@app.get("/task-status/{task_id}")
@api_key_required
def get_task_status(request: Request, task_id: str):
    task = tasks.get(task_id)
    if not task:
        return JSONResponse({"error": "Task not found"}, status_code=404)
    resp = {"task_id": task_id, "status": task["status"]}
    if task["status"] == "finished":
        resp["result"] = task["result"]
    if task["status"] == "failed":
        resp["error"] = task.get("error")
    return resp