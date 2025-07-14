import requests
import time
import os

API_URL = "https://protovideo-production.up.railway.app"
# API_URL = "http://localhost:8000"
API_KEY = "N8S6R_TydmHr58LoUzYZf9v2gRkcfWZemz1zWZ5WMkE"
HEADERS = {"x-api-key": API_KEY}

# Example payload (customize as needed)
payload = {
        "output_filename": "trading_motivation_shorts.mp4",
  "scenes": [
    {
      "type": "image",
      "subtitle": True,
      "promptImage": "A lone trader at dawn, silhouetted by soft morning light through a glass wall, city skyline in the background, glowing screens display candlestick charts and economic news",
      "narration_text": "Every trader wakes up to uncertainty, but the real battle begins not on the screen — it begins in the mind.",
      "duration": 5
    },
    {
      "type": "image",
      "subtitle": True,
      "promptImage": "A close-up of eyes staring intently at a fluctuating chart reflected in glasses, subtle sweat on the forehead, a ticking clock nearby",
      "narration_text": "In moments of volatility, fear whispers doubt and greed promises riches. Discipline must be louder than both.",
      "duration": 6
    },
    {
      "type": "image",
      "subtitle": True,
      "promptImage": "A trader’s journal open on a desk, filled with handwritten notes and emotional reflections, with a cup of coffee beside and a calm candle flickering",
      "narration_text": "The wise trader doesn’t just track entries and exits — they track emotions, thoughts, patterns within.",
      "duration": 6
    },
    {
      "type": "image",
      "subtitle": True,
      "promptImage": "A slow-motion shot of a trader stepping back from the desk, exhaling deeply, ambient blue light from monitors illuminating a peaceful face",
      "narration_text": "Mastery in trading isn’t about predicting the next move... it’s about remaining calm when chaos arrives.",
      "duration": 5
    },
    {
      "type": "image",
      "subtitle": True,
      "promptImage": "A serene outdoor scene with a trader walking in nature during sunset, earbuds in, listening to a podcast, distant sounds of nature replacing trading noise",
      "narration_text": "Step away when the mind clouds. Clarity returns in silence — not screens.",
      "duration": 5
    },
    {
      "type": "image",
      "subtitle": True,
      "promptImage": "A cinematic view of a seasoned trader watching past trades replayed on screen, learning, nodding, making new annotations",
      "narration_text": "Every loss is a mentor. Every setback — a step. In the journey of trading, growth is the real profit.",
      "duration": 6
    }
  ]
        }

def main():
    # 1. Start the task
    print("Sending /generate-task request...")
    resp = requests.post(f"{API_URL}/generate-task", json=payload, headers=HEADERS)
    resp.raise_for_status()
    data = resp.json()
    task_id = data["task_id"]
    # task_id="f6b3acc2-d48e-400c-a334-bb8c64e238a9"
    print(f"Task started: {task_id}")

    # 2. Poll for status every 30 seconds
    while True:
        print("Checking task status...")
        status_resp = requests.get(f"{API_URL}/task-status/{task_id}", headers=HEADERS)
        status_resp.raise_for_status()
        status_data = status_resp.json()
        print(f"Status: {status_data['status']}")
        print(status_data)
        if status_data["status"] == "finished":
            video_url = status_data["result"]["download_url"]
            print(f"Task finished! Downloading video from {video_url}")
            download_resp = requests.get(f"{API_URL}{video_url}", headers=HEADERS)
            download_resp.raise_for_status()
            out_path = os.path.join("results", f"api_task_{task_id}.mp4")
            os.makedirs("results", exist_ok=True)
            with open(out_path, "wb") as f:
                f.write(download_resp.content)
            print(f"Video downloaded to {out_path}")
            break
        elif status_data["status"] == "failed":
            print(f"Task failed: {status_data.get('error')}")
            break
        else:
            print("Task not finished yet. Waiting 30 seconds...")
            time.sleep(30)

if __name__ == "__main__":
    main() 