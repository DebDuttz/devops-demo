"""
Tests for main.py (ReAct Agent API).

The real `get_agent` dependency builds a ChatGroq client, which needs a live
GROQ_API_KEY and makes a network call. To keep these tests fast, free, and
offline, we override that dependency with a fake agent via FastAPI's
`app.dependency_overrides`.
"""
import os

# Safety net in case anything ever reads this at import time.
os.environ.setdefault("GROQ_API_KEY", "test-key")

import pytest
from fastapi.testclient import TestClient

from main import app, get_agent


class FakeAgentResult:
    """Mimics the object returned by ChatGroq.invoke() (has a .content attr)."""

    def __init__(self, content):
        self.content = content


class FakeAgent:
    """Drop-in replacement for the ChatGroq agent used in tests."""

    def __init__(self, response_text="This is a test answer."):
        self.response_text = response_text
        self.last_question = None

    def invoke(self, question):
        self.last_question = question
        return FakeAgentResult(self.response_text)


@pytest.fixture
def fake_agent():
    return FakeAgent()


@pytest.fixture
def client(fake_agent):
    app.dependency_overrides[get_agent] = lambda: fake_agent
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_ask_returns_200_and_expected_answer(client):
    response = client.post("/ask", json={"question": "What is DevOps?"})
    assert response.status_code == 200
    assert response.json() == {"answer": "This is a test answer."}


def test_ask_passes_question_through_to_agent(client, fake_agent):
    client.post("/ask", json={"question": "Explain CI/CD in one line"})
    assert fake_agent.last_question == "Explain CI/CD in one line"


def test_ask_reflects_agent_response(client, fake_agent):
    fake_agent.response_text = "Custom canned response"
    response = client.post("/ask", json={"question": "anything"})
    assert response.json()["answer"] == "Custom canned response"


def test_ask_missing_question_field_returns_422(client):
    response = client.post("/ask", json={})
    assert response.status_code == 422


def test_ask_wrong_type_for_question_returns_422(client):
    response = client.post("/ask", json={"question": 12345})
    assert response.status_code == 422


def test_ask_empty_string_question_is_accepted(client):
    response = client.post("/ask", json={"question": ""})
    assert response.status_code == 200


def test_ask_rejects_get_requests(client):
    response = client.get("/ask")
    assert response.status_code == 405


def test_app_metadata():
    assert app.title == "ReAct Agent API"
    assert app.version == "1.0.0"
