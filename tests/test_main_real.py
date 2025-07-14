# Copied and adapted from test_main.py
import pytest
from main import app
from fastapi.testclient import TestClient

client = TestClient(app)

def test_root_real():
    response = client.get("/")
    assert response.status_code == 200
    assert "Welcome" in response.json()["message"]

# Add more real endpoint tests as in the original test_main.py 