"""Shared test fixtures.

The OpenAI API is never called: `FakeToolCallingChatModel` replays a scripted
list of `AIMessage`s, including tool calls, so the whole agent loop — tool
selection, tool execution, final answer — runs offline and for free.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from app.agent import SupportAgent
from app.api_client import CustomerAPIClient
from app.config import get_settings
from app.knowledge_base import get_knowledge_base
from app.main import app, get_agent
from app.memory import conversation_memory
from app.mock_backend import load_customers


class FakeToolCallingChatModel(BaseChatModel):
    """A chat model that replays scripted responses instead of calling OpenAI.

    Attributes:
        responses: Messages to return, one per model invocation.
        error: If set, raised instead of responding (simulates an API failure).
        received: Every message list the model was invoked with, for assertions.
        bound_tools: Names of the tools that were bound to the model.
    """

    responses: list[AIMessage] = Field(default_factory=list)
    error: Exception | None = None
    received: list[list[BaseMessage]] = Field(default_factory=list)
    bound_tools: list[str] = Field(default_factory=list)
    index: int = 0

    model_config = {"arbitrary_types_allowed": True}

    @property
    def _llm_type(self) -> str:
        return "fake-tool-calling"

    def bind_tools(self, tools: Any, **kwargs: Any) -> BaseChatModel:
        """Record the tool names and return self — no real binding needed."""
        self.bound_tools = [getattr(t, "name", str(t)) for t in tools]
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.error is not None:
            raise self.error
        self.received.append(list(messages))
        if self.index >= len(self.responses):
            raise AssertionError(
                f"Fake model ran out of scripted responses (call #{self.index + 1})"
            )
        message = self.responses[self.index]
        self.index += 1
        return ChatResult(generations=[ChatGeneration(message=message)])


def tool_call_message(name: str, args: dict[str, Any] | None = None) -> AIMessage:
    """Build an assistant message that requests one tool call."""
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args or {}, "id": f"call_{name}"}],
    )


def final_message(text: str) -> AIMessage:
    """Build a plain assistant message with no tool calls."""
    return AIMessage(content=text)


@pytest.fixture(autouse=True)
def _isolate_state() -> Iterator[None]:
    """Reset cached settings, data and memory around every test."""
    get_settings.cache_clear()
    get_knowledge_base.cache_clear()
    load_customers.cache_clear()
    conversation_memory.reset()
    yield
    app.dependency_overrides.clear()
    get_settings.cache_clear()
    conversation_memory.reset()


@pytest.fixture
def client() -> Iterator[TestClient]:
    """A TestClient with the app's lifespan run."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def api_client() -> CustomerAPIClient:
    """A customer-API client that talks to the app in-process, not over TCP."""
    return CustomerAPIClient(
        base_url="http://backend.test",
        timeout=5.0,
        transport=httpx.ASGITransport(app=app),
    )


@pytest.fixture
def make_agent(api_client: CustomerAPIClient) -> Callable[..., SupportAgent]:
    """Factory building a `SupportAgent` backed by the scripted fake model."""

    def _make(
        *responses: AIMessage,
        error: Exception | None = None,
        client: CustomerAPIClient | None = None,
    ) -> SupportAgent:
        model = FakeToolCallingChatModel(responses=list(responses), error=error)
        return SupportAgent(chat_model=model, api_client=client or api_client)

    return _make


@pytest.fixture
def use_agent(client: TestClient) -> Callable[[SupportAgent], TestClient]:
    """Point the `/chat` endpoint at a specific agent instance."""

    def _use(agent: SupportAgent) -> TestClient:
        app.dependency_overrides[get_agent] = lambda: agent
        return client

    return _use


@pytest.fixture
def set_backend_failure(monkeypatch: pytest.MonkeyPatch) -> Callable[[str], None]:
    """Switch the mock backend into a failure mode for the current test."""

    def _set(mode: str) -> None:
        monkeypatch.setenv("BACKEND_FAILURE_MODE", mode)
        get_settings.cache_clear()

    return _set
