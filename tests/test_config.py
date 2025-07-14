import os
import importlib
import pytest

def test_config_defaults(monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    monkeypatch.delenv('R2_ENDPOINT_URL', raising=False)
    monkeypatch.delenv('PROTOVIDEO_API_KEY', raising=False)
    from video_generator.config import Config
    assert Config.OPENAI_API_KEY == ''
    assert Config.R2_ENDPOINT_URL == ''
    assert Config.API_KEY == 'N8S6R_TydmHr58LoUzYZf9v2gRkcfWZemz1zWZ5WMkE'

def test_config_env(monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'testkey')
    monkeypatch.setenv('R2_ENDPOINT_URL', 'http://test')
    monkeypatch.setenv('PROTOVIDEO_API_KEY', 'abc123')
    importlib.reload(__import__('video_generator.config'))
    from video_generator.config import Config
    assert Config.OPENAI_API_KEY == 'testkey'
    assert Config.R2_ENDPOINT_URL == 'http://test'
    assert Config.API_KEY == 'abc123' 