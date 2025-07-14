"""
Core video generation logic for ProtoVideo.
Handles orchestration, scene rendering, and helpers.
"""
from typing import List, Optional, Dict, Any
import os
import uuid
import tempfile
import logging
import gc
from moviepy import (
    ImageClip, VideoFileClip, AudioFileClip, concatenate_videoclips, CompositeVideoClip, TextClip, CompositeAudioClip
)
from moviepy.video.fx import CrossFadeIn, CrossFadeOut, MultiplyColor
from moviepy.audio.AudioClip import AudioClip, concatenate_audioclips
from .image_utils import download_asset, generate_image_from_prompt
from .audio_utils import generate_narration, generate_whisper_phrase_subtitles
from .cleanup_utils import cleanup_files
from .logging_utils import get_logger
from pydantic import BaseModel
from .config import Config

logger = get_logger()

TEMP_DIR = Config.TEMP_DIR
REEL_SIZE = Config.REEL_SIZE

class TextOverlay(BaseModel):
    content: str
    position: str = "center"
    fontsize: int = 36
    color: str = "white"

class SceneInput(BaseModel):
    type: str
    image: Optional[str] = None
    promptImage: Optional[str] = None
    video: Optional[str] = None
    narration: Optional[str] = None
    narration_text: Optional[str] = None
    music: Optional[str] = None
    duration: int
    text: Optional[TextOverlay] = None
    subtitle: bool = False

def render_scene(scene: SceneInput, use_global_narration: bool = False, task_id: Optional[str] = None) -> (str, List[str]):
    """
    Render a single scene (image or video) with optional narration, music, and subtitles.
    Returns the path to the rendered scene video and a list of temp files to clean up.
    """
    log_prefix = f"[task_id={task_id}] " if task_id else ""
    logger.info(f"{log_prefix}Rendering scene: {scene}")
    temp_files = []
    video_clip = None
    audio_clips = []
    # Handle image or video
    if scene.type == "image":
        image_path = None
        if scene.image:
            image_path = download_asset(scene.image)
            temp_files.append(image_path)
            logger.info(f"{log_prefix}Added image from file: {image_path}")
        elif scene.promptImage:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                logger.error(f"{log_prefix}OPENAI_API_KEY environment variable not set.")
                raise RuntimeError("OPENAI_API_KEY environment variable not set.")
            out_path = os.path.join(tempfile.gettempdir(), f"generated_{uuid.uuid4().hex}.png")
            try:
                image_path = generate_image_from_prompt(scene.promptImage, api_key, out_path)
                temp_files.append(image_path)
                logger.info(f"{log_prefix}Generated image from prompt: {image_path}")
            except Exception as e:
                logger.error(f"{log_prefix}Image generation failed: {e}")
                raise
        else:
            logger.error(f"{log_prefix}Image URL/path or promptImage required for image scene.")
            raise ValueError(f"{log_prefix}Image URL/path or promptImage required for image scene.")
        # --- Set duration from narration_text if present ---
        narration_path = None
        narration_audio = None
        if not use_global_narration:
            if scene.narration:
                narration_path = download_asset(scene.narration)
                temp_files.append(narration_path)
                logger.info(f"{log_prefix}Added narration from file: {narration_path}")
            elif scene.narration_text:
                narration_path = generate_narration(scene.narration_text)
                temp_files.append(narration_path)
                logger.info(f"{log_prefix}Added narration from text: {narration_path}")
                narration_audio = AudioFileClip(narration_path)
                silence = AudioClip(lambda t: 0, duration=1.5, fps=44100)
                scene.duration = narration_audio.duration + 1.5
        duration = scene.duration
        video_clip = ImageClip(image_path).with_duration(duration)
        video_clip = video_clip.resized(height=REEL_SIZE[1])
        if video_clip.w > REEL_SIZE[0]:
            video_clip = video_clip.resized(width=REEL_SIZE[0])
        def zoom(t):
            return 1.0 + 0.5 * (t / duration)
        video_clip = video_clip.resized(zoom)
        video_clip = video_clip.with_background_color(size=REEL_SIZE, color=(0,0,0), pos='center')
        video_clip = video_clip.with_effects([MultiplyColor(0.5)])
        # Add narration audio
        if not use_global_narration:
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
        # Add per-scene subtitles if requested
        if (
            getattr(scene, 'subtitle', False)
            and narration_path
            and scene.narration_text
        ):
            try:
                subtitle_clips = generate_whisper_phrase_subtitles(
                    narration_path, video_clip, words_per_line=4, font_size=50
                )
                video_clip = CompositeVideoClip([video_clip] + subtitle_clips)
                logger.info(f"{log_prefix}Subtitles added for scene narration.")
            except Exception as e:
                logger.warning(f"{log_prefix}Subtitle generation failed for scene: {e}")
        # Handle music
        if scene.music:
            music_path = download_asset(scene.music)
            temp_files.append(music_path)
            audio_clips.append(AudioFileClip(music_path).with_volume_scaled(0.3).with_duration(scene.duration))
            logger.info(f"{log_prefix}Added background music: {music_path}")
        # Mix audio
        if audio_clips:
            logger.info(f"{log_prefix}Mixing {len(audio_clips)} audio tracks for scene.")
            composite_audio = CompositeAudioClip(audio_clips)
            video_clip = video_clip.with_audio(composite_audio)
        scene_output = os.path.join(TEMP_DIR, f"scene_{uuid.uuid4().hex}.mp4")
        logger.info(f"{log_prefix}Exporting scene to {scene_output}")
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
        del video_clip
        gc.collect()
        logger.info(f"{log_prefix}Scene rendered and saved: {scene_output}")
        return scene_output, temp_files
    # TODO: Handle video scenes if needed
    raise NotImplementedError("Video scenes not implemented in refactor.") 