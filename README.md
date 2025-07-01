# Video Generation API

This service provides an API to generate short videos from a JSON input describing scenes, narration, music, and overlays.

## Features
- Accepts a POST request with a JSON body describing a video project
- Supports images, video clips, narration (MP3 or text-to-speech), background music, and text overlays
- Downloads remote assets and cleans up temp files
- Returns a download link for the generated video

## Requirements
- Python 3.10+
- Dependencies: `fastapi`, `uvicorn`, `moviepy`, `gtts`, `requests`, `aiofiles`, `python-multipart`

## Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install fastapi uvicorn moviepy gtts aiofiles python-multipart requests
```

## Running the API
```bash
uvicorn main:app --reload
```

## Example JSON Input
POST to `/generate` with:
```json
{
  "output_filename": "my_video.mp4",
  "scenes": [
    {
      "type": "image",
      "image": "https://example.com/image1.jpg",
      "narration": "https://example.com/narration1.mp3",
      "music": "https://example.com/music1.mp3",
      "duration": 5,
      "text": {
        "content": "Welcome to Remote Work",
        "position": "center",
        "fontsize": 48,
        "color": "white"
      }
    },
    {
      "type": "video",
      "video": "https://example.com/scene2.mp4",
      "narration_text": "This gives you more flexibility and time.",
      "duration": 7,
      "text": {
        "content": "More Freedom",
        "position": "bottom",
        "fontsize": 36,
        "color": "yellow"
      }
    }
  ]
}
```

## Endpoints
- `POST /generate` - Generate a video from the provided JSON
- `GET /download/{filename}` - Download the generated video
- `GET /` - API info and usage

## Notes
- All temp files are cleaned up after rendering.
- If narration_text is provided, gTTS will generate the narration audio.
- Error handling is included for missing/invalid assets. 