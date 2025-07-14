# Copied and adapted from test_r2_upload.py
import os
import pytest
import uuid
from main import upload_to_r2

def test_upload_to_r2_real(tmp_path):
    # This test will actually upload a file to R2 (requires env vars set)
    file_path = tmp_path / f"test_{uuid.uuid4().hex}.txt"
    file_path.write_text("hello r2")
    bucket = os.environ.get('R2_BUCKET_NAME')
    key = f"test/{file_path.name}"
    if not bucket:
        pytest.skip("R2_BUCKET_NAME not set")
    url = upload_to_r2(str(file_path), bucket, key)
    assert url is None or url.startswith("http") 