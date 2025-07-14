# Copied and adapted from test_audio.py
import pytest
from video_generator.audio_utils import generate_narration

def test_generate_narration_real():
    # This will actually generate audio (requires TTS model and dependencies)
    result = generate_narration("Hello world")
    assert result.endswith('.mp3') 