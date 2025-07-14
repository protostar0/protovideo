import os
import uuid
import pytest
from pathlib import Path
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert "/generate" in str(response.json())

def test_generate_invalid_scene_type():
    payload = {
        "output_filename": "test.mp4",
        "scenes": [
            {"type": "unknown", "duration": 2}
        ]
    }
    response = client.post("/generate", json=payload)
    assert response.status_code == 400
    assert "Unknown scene type" in response.text

def test_generate_missing_required_fields():
    payload = {"output_filename": "test.mp4", "scenes": [{}]}
    response = client.post("/generate", json=payload)
    assert response.status_code == 400 or response.status_code == 422

def test_generate_image_scene_with_invalid_url():
    payload = {
        "output_filename": "test.mp4",
        "scenes": [
            {"type": "image", "image": "http://invalid-url/doesnotexist.jpg", "duration": 2}
        ]
    }
    response = client.post("/generate", json=payload)
    assert response.status_code == 400
    assert "Failed to download asset" in response.text or "Asset not found" in response.text

def test_generate_real_image_scene(tmp_path):
    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a9/Example.jpg/320px-Example.jpg"
    unique_filename = f"trading_psychology_mindset_{uuid.uuid4().hex}.mp4"
    payload = {
        "output_filename": "trading_mindset_reel.mp4",
        "narration_text": "Success in trading doesn’t come from knowing every chart pattern or market signal. It comes from controlling your emotions, staying calm under pressure, and making disciplined decisions — even when everything feels uncertain. The market is unpredictable, but your mindset doesn’t have to be. When others panic, breathe. When others chase, wait. The traders who win long-term aren’t the ones who react — they’re the ones who plan, stay focused, and execute with confidence. Master your emotions, trust your process, and let your mindset become your greatest edge.",
        "scenes": [
            {"type": "image", "image": "https://images.pexels.com/photos/30572214/pexels-photo-30572214.jpeg", "duration": 6},
            {"type": "image", "image": "https://images.pexels.com/photos/17977092/pexels-photo-17977092.jpeg", "duration": 6},
            {"type": "image", "image": "https://images.pexels.com/photos/30572264/pexels-photo-30572264.jpeg", "duration": 6},
            {"type": "image", "image": "https://images.pexels.com/photos/14751274/pexels-photo-14751274.jpeg", "duration": 6}
        ]
    }
    response = client.post("/generate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "download_url" in data
    download_url = data["download_url"]
    download_response = client.get(download_url)
    assert download_response.status_code == 200
    assert download_response.headers["content-type"] == "video/mp4"
    video_path = tmp_path / f"test_real_video_{uuid.uuid4().hex}.mp4"
    with open(video_path, "wb") as f:
        f.write(download_response.content)
    assert os.path.getsize(video_path) > 1000

def test_generate_tempimage_scene(tmp_path):
    unique_filename = f"promptimage_test_{uuid.uuid4().hex}.mp4"
    payload = {
        "output_filename": "trading_motivation_shorts.mp4",
        "scenes": [
            {"type": "image", "subtitle": True, "image": "https://images.pexels.com/photos/30572214/pexels-photo-30572214.jpeg", "narration_text": "Trading psychology is the invisible force behind every decision you make.", "duration": 4, "subtitle": True},
            {"type": "image", "image": "https://images.pexels.com/photos/17977092/pexels-photo-17977092.jpeg", "narration_text": "Fear and greed can cloud your judgment. Recognize them, don’t suppress them.", "duration": 5, "subtitle": True}
        ]
    }
    response = client.post("/generate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "download_url" in data
    download_url = data["download_url"]
    download_response = client.get(download_url)
    assert download_response.status_code == 200
    assert download_response.headers["content-type"] == "video/mp4"
    video_path = tmp_path / f"test_prompttmpimage_video_{uuid.uuid4().hex}.mp4"
    with open(video_path, "wb") as f:
        f.write(download_response.content)
    assert os.path.getsize(video_path) > 1000

def test_generate_promptimage_scene(tmp_path):
    unique_filename = f"promptimage_test_{uuid.uuid4().hex}.mp4"
    payload = {
        "output_filename": "trading_motivation_shorts.mp4",
        "scenes": [
            {"type": "image", "subtitle": True, "promptImage": "A lone trader at dawn, silhouetted by soft morning light through a glass wall, city skyline in the background, glowing screens display candlestick charts and economic news", "narration_text": "Every trader wakes up to uncertainty, but the real battle begins not on the screen — it begins in the mind.", "duration": 5},
            {"type": "image", "subtitle": True, "promptImage": "A close-up of eyes staring intently at a fluctuating chart reflected in glasses, subtle sweat on the forehead, a ticking clock nearby", "narration_text": "In moments of volatility, fear whispers doubt and greed promises riches. Discipline must be louder than both.", "duration": 6},
            {"type": "image", "subtitle": True, "promptImage": "A trader’s journal open on a desk, filled with handwritten notes and emotional reflections, with a cup of coffee beside and a calm candle flickering", "narration_text": "The wise trader doesn’t just track entries and exits — they track emotions, thoughts, patterns within.", "duration": 6},
            {"type": "image", "subtitle": True, "promptImage": "A slow-motion shot of a trader stepping back from the desk, exhaling deeply, ambient blue light from monitors illuminating a peaceful face", "narration_text": "Mastery in trading isn’t about predicting the next move... it’s about remaining calm when chaos arrives.", "duration": 5},
            {"type": "image", "subtitle": True, "promptImage": "A serene outdoor scene with a trader walking in nature during sunset, earbuds in, listening to a podcast, distant sounds of nature replacing trading noise", "narration_text": "Step away when the mind clouds. Clarity returns in silence — not screens.", "duration": 5},
            {"type": "image", "subtitle": True, "promptImage": "A cinematic view of a seasoned trader watching past trades replayed on screen, learning, nodding, making new annotations", "narration_text": "Every loss is a mentor. Every setback — a step. In the journey of trading, growth is the real profit.", "duration": 6}
        ]
    }
    response = client.post("/generate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "download_url" in data
    download_url = data["download_url"]
    download_response = client.get(download_url)
    assert download_response.status_code == 200
    assert download_response.headers["content-type"] == "video/mp4"
    video_path = tmp_path / f"test_promptimage_video_{uuid.uuid4().hex}.mp4"
    with open(video_path, "wb") as f:
        f.write(download_response.content)
    assert os.path.getsize(video_path) > 1000 