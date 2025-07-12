import os
from main import upload_to_r2

if __name__ == "__main__":
    # Example: upload the first .mp4 file found in the results directory
    results_dir = "results"
    files = [f for f in os.listdir(results_dir) if f.endswith('.mp4')]
    if not files:
        print("No .mp4 files found in results directory.")
        exit(1)
    video_file = os.path.join(results_dir, files[0])
    bucket_name = os.environ.get('R2_BUCKET_NAME')
    if not bucket_name:
        print("R2_BUCKET_NAME environment variable not set.")
        exit(1)
    object_key = os.path.basename(video_file)
    r2_url = upload_to_r2(video_file, bucket_name, object_key)
    print(f"R2 URL: {r2_url}") 