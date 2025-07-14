import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_generate_task_missing_key():
    # Should fail without API key
    response = client.post("/generate-task", json={"output_filename": "test.mp4", "scenes": []})
    assert response.status_code == 401

def test_generate_task_with_key(monkeypatch):
    # Should queue a task with correct API key
    payload = {"output_filename": "test.mp4", "scenes": []}
    headers = {"x-api-key": "N8S6R_TydmHr58LoUzYZf9v2gRkcfWZemz1zWZ5WMkE"}
    # Mock the thread start to avoid actually running the worker
    monkeypatch.setattr("threading.Thread.start", lambda self: None)
    response = client.post("/generate-task", json=payload, headers=headers)
    assert response.status_code == 200
    assert "task_id" in response.json() 