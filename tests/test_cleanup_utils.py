import os
import pytest
from video_generator import cleanup_utils

def test_cleanup_files(tmp_path):
    # Create temp files
    file1 = tmp_path / 'file1.txt'
    file2 = tmp_path / 'file2.txt'
    file1.write_text('a')
    file2.write_text('b')
    assert file1.exists() and file2.exists()
    cleanup_utils.cleanup_files([str(file1), str(file2)])
    assert not file1.exists() and not file2.exists()
    # Should not raise if file does not exist
    cleanup_utils.cleanup_files([str(file1)]) 