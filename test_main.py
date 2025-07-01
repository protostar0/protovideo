import os
import tempfile
from fastapi.testclient import TestClient
from main import app
from pathlib import Path

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
    # Use a public domain image (e.g., Wikimedia Commons)
    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a9/Example.jpg/320px-Example.jpg"
    payload = {
  "output_filename": "trading_psychology_mindset.mp4",
  "scenes": [
    {
      "type": "image",
      "image": "https://images.pexels.com/photos/210607/pexels-photo-210607.jpeg",
      "narration_text": "Success in trading begins with mastering your own mind. Emotions can cloud judgment and lead to impulsive decisions.",
      "duration": 6,
      "text": {
        "content": "Master Your Mind",
        "position": "center",
        "fontsize": 48,
        "color": "white"
      }
    },    {
      "type": "image",
      "image": "https://pixabay.com/get/g6629f1c191a40528b5d3c52ecd480f03c1ebf8e1283c5ebd88f9901136da5f50d1b427bf7a6edaeb4bb689e429f1b072adfdb750f417803e4978430d1b0c47dc_1280.jpg",
      "narration_text": "Discipline and patience are key traits of successful traders. Stick to your strategy, even when the market tests your resolve.",
      "duration": 6,
      "text": {
        "content": "Discipline Over Impulse",
        "position": "bottom",
        "fontsize": 44,
        "color": "yellow"
      }
    },
    {
      "type": "video",
      "video": "https://cdn.pixabay.com/video/2020/07/02/43607-436780299_medium.mp4",
      "narration_text": "A calm and focused mind navigates the volatile markets with clarity. Mindfulness can be your greatest asset.",
      "duration": 8,
      "text": {
        "content": "Stay Calm, Trade Smart",
        "position": "top",
        "fontsize": 42,
        "color": "lightblue"
      }
    },
  ]
}

    response = client.post("/generate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "download_url" in data
    # Download the video
    download_url = data["download_url"]
    download_response = client.get(download_url)
    assert download_response.status_code == 200
    assert download_response.headers["content-type"] == "video/mp4"
    # Save to temp file and check size
    video_path = tmp_path / "test_real_video2.mp4"
    with open(video_path, "wb") as f:
        f.write(download_response.content)
    assert os.path.getsize(video_path) > 1000  # Should be a non-trivial file

# Note: For a full integration test, you would need to provide valid URLs or local files for image/video/audio assets.
# This is a minimal set of tests for API structure and error handling. 
test_generate_real_image_scene(Path("./results/"))