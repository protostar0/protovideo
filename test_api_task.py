import requests
import time
import os

API_URL = "https://protovideo-production.up.railway.app"
API_KEY = "N8S6R_TydmHr58LoUzYZf9v2gRkcfWZemz1zWZ5WMkE"
HEADERS = {"x-api-key": API_KEY}

# Example payload (customize as needed)
payload = {
            "output_filename": "trading_mindset_reel.mp4",
                "narration_text": "Success in trading doesn’t come from knowing every chart pattern or market signal. It comes from controlling your emotions, staying calm under pressure, and making disciplined decisions — even when everything feels uncertain. The market is unpredictable, but your mindset doesn’t have to be. When others panic, breathe. When others chase, wait. The traders who win long-term aren’t the ones who react — they’re the ones who plan, stay focused, and execute with confidence. Master your emotions, trust your process, and let your mindset become your greatest edge.",
            "scenes": [
                {
                "type": "image",
                "image": "https://images.pexels.com/photos/30572214/pexels-photo-30572214.jpeg",
                "duration": 6
                },
                {
                "type": "image",
                "image": "https://images.pexels.com/photos/17977092/pexels-photo-17977092.jpeg",
                "duration": 6
                },
                {
                "type": "image",
                "image": "https://images.pexels.com/photos/30572264/pexels-photo-30572264.jpeg",
                "duration": 6
                },
                {
                "type": "image",
                "image": "https://images.pexels.com/photos/14751274/pexels-photo-14751274.jpeg",
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
    print(f"Task started: {task_id}")

    # 2. Poll for status every 30 seconds
    while True:
        print("Checking task status...")
        status_resp = requests.get(f"{API_URL}/task-status/{task_id}", headers=HEADERS)
        status_resp.raise_for_status()
        status_data = status_resp.json()
        print(f"Status: {status_data['status']}")
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