import openai
import requests
import os
import time

def generate_image_from_prompt(prompt, api_key, out_path="generated_image.png", retries=3, delay=5):
    """
    Generate an image from a text prompt using OpenAI's API and save it to out_path.
    Retries on failure.
    """
    openai.api_key = api_key
    last_exception = None
    for attempt in range(retries):
        try:
            response = openai.Image.create(
                prompt=prompt,
                n=1,
                size="1024x1024"
            )
            image_url = response['data'][0]['url']
            img_data = requests.get(image_url).content
            with open(out_path, 'wb') as handler:
                handler.write(img_data)
            return out_path
        except Exception as e:
            last_exception = e
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                raise RuntimeError(f"OpenAI image generation failed after {retries} attempts: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m video_generator.generate_image 'your prompt here' [output_path]")
        exit(1)
    prompt = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else "generated_image.png"
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Set OPENAI_API_KEY env variable.")
        exit(1)
    out_path = generate_image_from_prompt(prompt, api_key, out_path)
    print(f"Image saved to {out_path}") 