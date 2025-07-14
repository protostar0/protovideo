import pytest
from unittest import mock
from video_generator.generator import render_scene, SceneInput

def test_render_scene_image(monkeypatch, tmp_path):
    # Mock all external dependencies
    with mock.patch('video_generator.generator.download_asset') as download_asset, \
         mock.patch('video_generator.generator.generate_image_from_prompt') as gen_img, \
         mock.patch('video_generator.generator.generate_narration') as gen_narr, \
         mock.patch('video_generator.generator.AudioFileClip') as AudioFileClip, \
         mock.patch('video_generator.generator.ImageClip') as ImageClip, \
         mock.patch('video_generator.generator.CompositeAudioClip') as CompositeAudioClip, \
         mock.patch('video_generator.generator.CompositeVideoClip') as CompositeVideoClip:
        download_asset.side_effect = lambda x: '/tmp/fake.png' if x.endswith('.png') else '/tmp/fake.mp3'
        gen_img.return_value = '/tmp/fake.png'
        gen_narr.return_value = '/tmp/fake.mp3'
        audio_clip = mock.Mock()
        audio_clip.duration = 2.0
        AudioFileClip.return_value = audio_clip
        image_clip = mock.Mock()
        image_clip.with_duration.return_value = image_clip
        image_clip.resized.return_value = image_clip
        image_clip.with_background_color.return_value = image_clip
        image_clip.with_effects.return_value = image_clip
        image_clip.with_audio.return_value = image_clip
        ImageClip.return_value = image_clip
        CompositeAudioClip.return_value = image_clip
        CompositeVideoClip.return_value = image_clip
        image_clip.write_videofile.return_value = None
        image_clip.close.return_value = None
        scene = SceneInput(
            type='image',
            image='http://example.com/img.png',
            narration_text='hello',
            music='http://example.com/music.mp3',
            duration=3
        )
        out, temp_files = render_scene(scene)
        assert out.endswith('.mp4')
        assert isinstance(temp_files, list) 