"""The LangChain tool-calling agent that drives a support conversation.

Flow for one turn:

    message + history + signed-in customer
        -> ChatPromptTemplate (system rules + session context)
        -> ChatOpenAI with tools bound
        -> the model decides: knowledge base, account API, both, or neither
        -> AgentExecutor runs the chosen tools and feeds results back
        -> final natural-language answer + the tools that were used

Every failure of the model or the executor is caught here and turned into a
safe fallback answer, so `/chat` cannot return a 500 or a stack trace.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from app.api_client import CustomerAPIClient
from app.config import Settings, get_settings
from app.errors import LLMUnavailableError
from app.prompts import LLM_FALLBACK_ANSWER, SYSTEM_PROMPT, build_context_preamble
from app.tools import ToolContext, build_tools

logger = logging.getLogger(__name__)

# AgentExecutor emits this when it hits `max_iterations`; it is an internal
# detail, so we replace it with something a customer can act on.
_ITERATION_LIMIT_MARKER = "Agent stopped due to"

_ITERATION_LIMIT_ANSWER = (
    "I wasn't able to work that one out. Could you rephrase it, or ask about a "
    "single thing at a time? I can also connect you with a human agent."
)


@dataclass
class AgentResult:
    """The outcome of one agent turn."""

    answer: str
    tools_used: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    degraded: bool = False


def create_chat_model(settings: Settings | None = None) -> BaseChatModel:
    """Build the chat model from settings.

    Defaults to OpenAI. Setting `OPENAI_BASE_URL` points the same client at any
    OpenAI-compatible endpoint — Groq, OpenRouter, Together, or a local server —
    provided the chosen model supports tool calling, which this agent requires.

    Raises:
        LLMUnavailableError: if no API key is configured.
    """
    settings = settings or get_settings()
    if not settings.llm_configured:
        raise LLMUnavailableError("OPENAI_API_KEY is not set")

    options: dict[str, Any] = {
        "model": settings.openai_model,
        "temperature": settings.openai_temperature,
        "timeout": settings.openai_timeout_seconds,
        "max_retries": settings.openai_max_retries,
        "api_key": settings.openai_api_key,
    }
    if settings.openai_base_url:
        options["base_url"] = settings.openai_base_url
        logger.info("Using OpenAI-compatible endpoint at %s", settings.openai_base_url)
    return ChatOpenAI(**options)


def _build_prompt(customer_id: str | None) -> ChatPromptTemplate:
    """Assemble the prompt for one request.

    The system message is passed as a concrete `SystemMessage` rather than a
    template string so nothing inside the prompt text (or a customer id) is
    ever interpreted as a template variable.
    """
    system_text = f"{SYSTEM_PROMPT}\n{build_context_preamble(customer_id)}"
    return ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=system_text),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )


class SupportAgent:
    """Runs support conversations against a chat model and a set of tools.

    Args:
        chat_model: Any LangChain chat model that supports tool calling. Tests
            inject a scripted fake so no API credits are spent. When omitted,
            the configured OpenAI model is built on first use — building it
            lazily keeps a missing API key from pre-empting request validation.
        api_client: Client for the internal customer backend.
        max_iterations: Cap on tool-call rounds per turn.
    """

    def __init__(
        self,
        chat_model: BaseChatModel | None = None,
        api_client: CustomerAPIClient | None = None,
        max_iterations: int | None = None,
    ) -> None:
        self._chat_model = chat_model
        self._api_client = api_client
        self._max_iterations = max_iterations or get_settings().agent_max_iterations

    def _resolve_model(self) -> BaseChatModel:
        """Return the chat model, constructing the OpenAI one on first use.

        Raises:
            LLMUnavailableError: if no model was injected and no API key is set.
        """
        if self._chat_model is None:
            self._chat_model = create_chat_model()
        return self._chat_model

    async def answer(
        self,
        message: str,
        customer_id: str | None = None,
        history: list[BaseMessage] | None = None,
    ) -> AgentResult:
        """Answer one user message, calling tools if the model decides to.

        Raises:
            LLMUnavailableError: only when the model is not configured at all —
                a deployment problem the caller should see. Every runtime
                failure instead becomes a degraded but polite answer.
        """
        context = ToolContext(customer_id=customer_id)
        tools = build_tools(context, api_client=self._api_client)
        chat_model = self._resolve_model()

        try:
            agent = create_tool_calling_agent(
                llm=chat_model, tools=tools, prompt=_build_prompt(customer_id)
            )
            executor = AgentExecutor(
                agent=agent,
                tools=tools,
                max_iterations=self._max_iterations,
                return_intermediate_steps=True,
                handle_parsing_errors=True,
                verbose=False,
            )
            result = await executor.ainvoke(
                {"input": message, "chat_history": history or []}
            )
        except Exception:  # noqa: BLE001 - the model/provider is the only thing left
            logger.exception("Agent turn failed for customer_id=%s", customer_id)
            return AgentResult(answer=LLM_FALLBACK_ANSWER, degraded=True)

        answer = str(result.get("output") or "").strip()
        tools_used = _extract_tools_used(result.get("intermediate_steps") or [])

        if not answer or answer.startswith(_ITERATION_LIMIT_MARKER):
            logger.warning(
                "Agent produced no usable answer (tools_used=%s); returning fallback",
                tools_used,
            )
            return AgentResult(
                answer=_ITERATION_LIMIT_ANSWER,
                tools_used=tools_used,
                sources=context.sources,
                degraded=True,
            )

        return AgentResult(
            answer=answer,
            tools_used=tools_used,
            sources=context.sources,
            degraded=False,
        )


def _extract_tools_used(intermediate_steps: list) -> list[str]:
    """Pull the tool names out of AgentExecutor's intermediate steps, in order."""
    names: list[str] = []
    for step in intermediate_steps:
        action = step[0] if isinstance(step, (tuple, list)) and step else None
        name = getattr(action, "tool", None)
        if isinstance(name, str) and name not in names:
            names.append(name)
    return names
