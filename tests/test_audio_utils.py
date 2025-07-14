import pytest
from unittest import mock
from video_generator import audio_utils

def test_generate_narration():
    with mock.patch('video_generator.audio_utils.ChatterboxTTS') as m:
        instance = m.from_pretrained.return_value
        instance.generate.return_value = b'fakewav'
        instance.sr = 22050
        with mock.patch('video_generator.audio_utils.ta.save') as save_mock:
            save_mock.return_value = None
            result = audio_utils.generate_narration('hello')
            assert result.endswith('.mp3')

def test_generate_whisper_phrase_subtitles():
    with mock.patch('video_generator.audio_utils.whisper.load_model') as m:
        model = m.return_value
        model.transcribe.return_value = {
            'segments': [
                {'words': [
                    {'word': 'hello', 'start': 0, 'end': 1},
                    {'word': 'world', 'start': 1, 'end': 2}
                ]}
            ]
        }
        class DummyClip:
            w = 500
            h = 1000
        result = audio_utils.generate_whisper_phrase_subtitles('audio.wav', DummyClip(), words_per_line=1, font_size=10)
        assert len(result) == 2 