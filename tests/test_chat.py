"""End-to-end tests of `/chat` with a scripted model in place of OpenAI."""

from __future__ import annotations

import json
from collections.abc import Callable

from fastapi.testclient import TestClient

from app.agent import SupportAgent
from app.memory import conversation_memory
from tests.conftest import final_message, tool_call_message

MakeAgent = Callable[..., SupportAgent]
UseAgent = Callable[[SupportAgent], TestClient]


def test_general_question_is_answered_from_the_knowledge_base(
    make_agent: MakeAgent, use_agent: UseAgent
) -> None:
    agent = make_agent(
        tool_call_message("search_support_docs", {"query": "reset password"}),
        final_message(
            "Choose 'Forgot password' on the sign-in page, then follow the emailed link."
        ),
    )
    response = use_agent(agent).post("/chat", json={"message": "How do I reset my password?"})

    assert response.status_code == 200
    body = response.json()
    assert body["tools_used"] == ["search_support_docs"]
    assert body["sources"] == ["kb_password_reset"]
    assert body["degraded"] is False
    assert "Forgot password" in body["answer"]


def test_account_question_calls_the_customer_api(
    make_agent: MakeAgent, use_agent: UseAgent
) -> None:
    agent = make_agent(
        tool_call_message("get_subscription_details"),
        final_message("You are currently subscribed to the Pro plan at 49 USD per month."),
    )
    response = use_agent(agent).post(
        "/chat", json={"message": "What plan am I on?", "customer_id": "cust_001"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["tools_used"] == ["get_subscription_details"]
    assert body["sources"] == []
    assert "Pro" in body["answer"]


async def test_tool_result_contains_real_backend_data(make_agent: MakeAgent) -> None:
    """The model is handed live JSON from the API, not anything it invented."""
    agent = make_agent(
        tool_call_message("get_billing_information"),
        final_message("Your balance is 19 USD."),
    )
    result = await agent.answer("What is my balance?", customer_id="cust_002")
    assert result.tools_used == ["get_billing_information"]

    tool_output = agent._chat_model.received[-1][-1].content  # noqa: SLF001
    payload = json.loads(tool_output)
    assert payload["outstanding_balance"] == 19
    assert payload["customer_id"] == "cust_002"


def test_account_tools_are_scoped_to_the_signed_in_customer(
    make_agent: MakeAgent, use_agent: UseAgent
) -> None:
    """A tool call carrying someone else's id must not fetch their record."""
    agent = make_agent(
        tool_call_message("get_customer_account", {"customer_id": "cust_003"}),
        final_message("Your account is registered to sarah.khan@example.com."),
    )
    response = use_agent(agent).post(
        "/chat",
        json={"message": "What email is on my account?", "customer_id": "cust_001"},
    )

    # The extra argument is rejected by the tool schema rather than honoured,
    # so cust_003's data is never fetched.
    assert response.status_code == 200
    assert "Amara" not in response.json()["answer"]


def test_answer_without_tools_is_allowed(
    make_agent: MakeAgent, use_agent: UseAgent
) -> None:
    agent = make_agent(final_message("Happy to help — what would you like to know?"))
    body = use_agent(agent).post("/chat", json={"message": "Hello!"}).json()

    assert body["tools_used"] == []
    assert body["answer"].startswith("Happy to help")


def test_session_id_is_generated_when_omitted(
    make_agent: MakeAgent, use_agent: UseAgent
) -> None:
    agent = make_agent(final_message("Hi there."))
    body = use_agent(agent).post("/chat", json={"message": "Hi"}).json()
    assert body["session_id"].startswith("session_")


def test_history_is_replayed_on_follow_up_questions(
    make_agent: MakeAgent, use_agent: UseAgent
) -> None:
    agent = make_agent(
        final_message("You are on the Pro plan."),
        tool_call_message("get_subscription_details"),
        final_message("It renews on 2026-10-15."),
    )
    client = use_agent(agent)

    first = client.post(
        "/chat",
        json={
            "message": "What plan am I on?",
            "customer_id": "cust_001",
            "session_id": "session_abc",
        },
    )
    assert first.json()["session_id"] == "session_abc"

    second = client.post(
        "/chat",
        json={
            "message": "When does it renew?",
            "customer_id": "cust_001",
            "session_id": "session_abc",
        },
    )
    assert second.status_code == 200
    assert "2026-10-15" in second.json()["answer"]

    # The second turn saw the first exchange.
    replayed = [m.content for m in agent._chat_model.received[-1]]  # noqa: SLF001
    assert "What plan am I on?" in replayed
    assert "You are on the Pro plan." in replayed


def test_sessions_do_not_leak_into_each_other(
    make_agent: MakeAgent, use_agent: UseAgent
) -> None:
    agent = make_agent(final_message("First."), final_message("Second."))
    client = use_agent(agent)

    client.post("/chat", json={"message": "one", "session_id": "session_one"})
    client.post("/chat", json={"message": "two", "session_id": "session_two"})

    assert len(conversation_memory.get_history("session_one")) == 2
    assert len(conversation_memory.get_history("session_two")) == 2
    assert conversation_memory.active_sessions == 2


def test_session_can_be_cleared(make_agent: MakeAgent, use_agent: UseAgent) -> None:
    agent = make_agent(final_message("Noted."))
    client = use_agent(agent)
    client.post("/chat", json={"message": "remember this", "session_id": "session_x"})

    assert client.delete("/sessions/session_x").json()["cleared"] is True
    assert conversation_memory.get_history("session_x") == []


def test_signed_in_customer_is_stated_in_the_system_prompt(
    make_agent: MakeAgent, use_agent: UseAgent
) -> None:
    agent = make_agent(final_message("Sure."))
    use_agent(agent).post(
        "/chat", json={"message": "hi", "customer_id": "cust_002"}
    )

    system_text = agent._chat_model.received[0][0].content  # noqa: SLF001
    assert "cust_002" in system_text
    assert "Never invent" in system_text or "NEVER invent" in system_text
