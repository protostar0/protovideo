"""
Audio utilities for ProtoVideo.
Handles narration generation and Whisper-based subtitle generation.
"""
from typing import Optional, List
import os
import uuid
import logging
from fastapi import HTTPException
from gtts import gTTS
from moviepy.audio.io.AudioFileClip import AudioFileClip
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
import whisper
from moviepy import TextClip, CompositeVideoClip
from .config import Config

def generate_narration(text: str) -> str:
    """
    Generate narration audio from text using ChatterboxTTS.
    Returns the path to the generated audio file.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Generating narration for text: {text[:60]}...")
    TEMP_DIR = Config.TEMP_DIR
    local_filename = os.path.join(TEMP_DIR, f"narration_{uuid.uuid4().hex}.mp3")
    try:
        model = ChatterboxTTS.from_pretrained(device="cpu")
        wav = model.generate(
            text,
            exaggeration=0.5,
            cfg_weight=0.5
        )
        ta.save(local_filename, wav, model.sr)
        logger.info(f"Generated narration at {local_filename}")
        return local_filename
    except Exception as e:
        logger.error(f"Failed to generate narration: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate narration: {e}")

def generate_whisper_phrase_subtitles(audio_path: str, video_clip, words_per_line: int = 4, font_size: int = 50) -> List:
    """
    Generate animated phrase subtitles using Whisper for a given audio file and video clip.
    Returns a list of subtitle TextClip objects.
    """
    import traceback
    logger = logging.getLogger(__name__)
    logger.info(f"[DEBUG] Entered generate_whisper_phrase_subtitles with audio_path={audio_path}, video_clip={video_clip}, words_per_line={words_per_line}, font_size={font_size}")
    model = whisper.load_model("base")
    try:
        result = model.transcribe(audio_path, word_timestamps=True, verbose=False)
    except Exception as e:
        logger.error(f"[DEBUG] Exception during whisper transcribe: {e}\n{traceback.format_exc()}")
        raise
    all_words = []
    for segment in result['segments']:
        all_words.extend(segment.get('words', []))
    lines = []
    for i in range(0, len(all_words), words_per_line):
        chunk = all_words[i:i + words_per_line]
        if not chunk:
            continue
        line_text = ' '.join([w['word'].strip() for w in chunk])
        start = chunk[0]['start']
        end = chunk[-1]['end']
        lines.append({'text': line_text.upper(), 'start': start, 'end': end, 'words': chunk})
    subtitle_clips = []
    for line in lines:
        try:
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
                .with_position(("center", int(video_clip.h * 0.6)))
                .with_start(line['start'])
                .with_duration(line['end'] - line['start'])
            )
            subtitle_clips.append(base_clip)
        except Exception as e:
            logger.error(f"[DEBUG] Exception during subtitle clip creation: {e}\n{traceback.format_exc()}")
            raise
    return subtitle_clips 