"""Tests for every failure path: bad input, dead backend, dead model.

The invariant under test throughout: the client gets a clear, safe message and
the process stays up. No stack traces, no provider errors, no invented data.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest
from fastapi.testclient import TestClient

from app.agent import SupportAgent
from app.api_client import CustomerAPIClient
from app.errors import BackendUnavailableError, MalformedBackendResponseError
from app.memory import conversation_memory
from tests.conftest import final_message, tool_call_message

MakeAgent = Callable[..., SupportAgent]
UseAgent = Callable[[SupportAgent], TestClient]

LEAK_MARKERS = ("Traceback", "openai", "httpx", "OPENAI_API_KEY", "127.0.0.1", "app/")


def assert_no_internals(text: str) -> None:
    """Fail if a response body looks like it leaked implementation detail."""
    for marker in LEAK_MARKERS:
        assert marker not in text, f"response leaked {marker!r}: {text}"


# ---------------------------------------------------------------------------
# Request validation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "payload",
    [
        {},  # message missing
        {"message": ""},  # message empty
        {"message": "   "},  # message blank
        {"message": "hi", "customer_id": "'; DROP TABLE customers;--"},
        {"message": "hi", "customer_id": "../../etc/passwd"},
        {"message": "hi", "customer_id": "cust_001 OR 1=1"},
        {"message": "hi", "session_id": "bad session id!"},
        {"message": "x" * 2001},  # over the length cap
    ],
)
def test_invalid_requests_are_rejected_with_422(
    client: TestClient, payload: dict
) -> None:
    response = client.post("/chat", json=payload)
    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "invalid_request"
    assert_no_internals(json.dumps(body))


def test_unknown_customer_id_shape_never_reaches_the_agent(client: TestClient) -> None:
    """Validation runs before the agent, so no model call is made at all."""
    response = client.post("/chat", json={"message": "hi", "customer_id": "admin"})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Backend failures
# ---------------------------------------------------------------------------
async def test_backend_timeout_becomes_a_typed_error() -> None:
    """A read timeout maps to BackendUnavailableError, not an httpx exception."""

    def time_out(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    timing_out = CustomerAPIClient(
        base_url="http://backend.test",
        timeout=0.1,
        transport=httpx.MockTransport(time_out),
    )
    with pytest.raises(BackendUnavailableError):
        await timing_out.get_customer("cust_001")


async def test_backend_server_error_becomes_a_typed_error(
    api_client: CustomerAPIClient, set_backend_failure: Callable[[str], None]
) -> None:
    set_backend_failure("server_error")
    with pytest.raises(BackendUnavailableError):
        await api_client.get_billing("cust_001")


async def test_malformed_backend_json_becomes_a_typed_error(
    api_client: CustomerAPIClient, set_backend_failure: Callable[[str], None]
) -> None:
    set_backend_failure("malformed")
    with pytest.raises(MalformedBackendResponseError):
        await api_client.get_customer("cust_001")


async def test_unreachable_backend_becomes_a_typed_error() -> None:
    """A connection refused on a real socket, not a simulated failure."""
    unreachable = CustomerAPIClient(base_url="http://127.0.0.1:9", timeout=1.0)
    with pytest.raises(BackendUnavailableError):
        await unreachable.get_customer("cust_001")


def test_tools_report_backend_failure_instead_of_raising(
    make_agent: MakeAgent,
    use_agent: UseAgent,
    set_backend_failure: Callable[[str], None],
) -> None:
    """The model is told the lookup failed, and answers from that — not a guess."""
    set_backend_failure("server_error")
    agent = make_agent(
        tool_call_message("get_billing_information"),
        final_message(
            "I'm unable to retrieve your billing information right now. "
            "Please try again in a few minutes."
        ),
    )
    response = use_agent(agent).post(
        "/chat", json={"message": "What is my balance?", "customer_id": "cust_002"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["tools_used"] == ["get_billing_information"]
    assert "unable to retrieve" in body["answer"]
    assert_no_internals(json.dumps(body))

    tool_output = json.loads(agent._chat_model.received[-1][-1].content)  # noqa: SLF001
    assert tool_output["error"] == "backend_unavailable"


def test_tools_report_a_missing_customer_instead_of_inventing_one(
    make_agent: MakeAgent, use_agent: UseAgent
) -> None:
    agent = make_agent(
        tool_call_message("get_customer_account"),
        final_message("I couldn't find an account with those details."),
    )
    response = use_agent(agent).post(
        "/chat", json={"message": "What is my plan?", "customer_id": "cust_777"}
    )

    assert response.status_code == 200
    tool_output = json.loads(agent._chat_model.received[-1][-1].content)  # noqa: SLF001
    assert tool_output["error"] == "customer_not_found"


def test_account_tools_refuse_to_run_without_a_signed_in_customer(
    make_agent: MakeAgent, use_agent: UseAgent
) -> None:
    agent = make_agent(
        tool_call_message("get_subscription_details"),
        final_message("Please sign in and I can check your plan."),
    )
    response = use_agent(agent).post("/chat", json={"message": "What plan am I on?"})

    assert response.status_code == 200
    tool_output = json.loads(agent._chat_model.received[-1][-1].content)  # noqa: SLF001
    assert tool_output["error"] == "not_signed_in"


# ---------------------------------------------------------------------------
# Model failures
# ---------------------------------------------------------------------------
def test_openai_failure_returns_a_safe_fallback_not_a_500(
    make_agent: MakeAgent, use_agent: UseAgent
) -> None:
    agent = make_agent(error=RuntimeError("connection reset by peer"))
    response = use_agent(agent).post("/chat", json={"message": "Hello"})

    assert response.status_code == 200
    body = response.json()
    assert body["degraded"] is True
    assert "trouble reaching" in body["answer"]
    assert_no_internals(json.dumps(body))


def test_degraded_turns_are_not_written_to_history(
    make_agent: MakeAgent, use_agent: UseAgent
) -> None:
    agent = make_agent(error=RuntimeError("boom"))
    use_agent(agent).post(
        "/chat", json={"message": "Hello", "session_id": "session_fail"}
    )
    assert conversation_memory.get_history("session_fail") == []


def test_missing_api_key_returns_503_with_a_safe_message(client: TestClient) -> None:
    """No dependency override here, so the real (unconfigured) agent is used."""
    response = client.post("/chat", json={"message": "Hello"})
    assert response.status_code == 503

    body = response.json()
    assert body["error"] == "llm_unavailable"
    assert_no_internals(json.dumps(body))


def test_iteration_limit_produces_a_graceful_answer(
    make_agent: MakeAgent, use_agent: UseAgent
) -> None:
    """A model stuck in a tool loop is stopped and answered for, not left hanging."""
    agent = make_agent(
        *[tool_call_message("search_support_docs", {"query": "refunds"})] * 8
    )
    response = use_agent(agent).post("/chat", json={"message": "Tell me about refunds"})

    assert response.status_code == 200
    body = response.json()
    assert body["degraded"] is True
    assert "Agent stopped" not in body["answer"]
    assert_no_internals(json.dumps(body))


def test_unknown_route_returns_structured_json(client: TestClient) -> None:
    response = client.get("/api/customers/cust_001/does-not-exist")
    assert response.status_code == 404
    assert "detail" in response.json()
