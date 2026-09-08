"""LangChain tools the agent can call.

Two design choices are worth calling out:

1. **Account tools take no customer argument.** The signed-in customer id is
   closed over when the tools are built for a request, so the model physically
   cannot point a lookup at somebody else's account, however it is prompted.
2. **Tools never raise.** Every failure is converted into a small JSON error
   object that the model can read and explain. An exception escaping into the
   agent loop would abort the turn; a structured error lets the assistant
   degrade gracefully instead.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.api_client import CustomerAPIClient
from app.errors import (
    BackendUnavailableError,
    CustomerNotFoundError,
    MalformedBackendResponseError,
    SupportAgentError,
)
from app.knowledge_base import get_knowledge_base

logger = logging.getLogger(__name__)

ACCOUNT_TOOL_NAMES = frozenset(
    {"get_customer_account", "get_subscription_details", "get_billing_information"}
)


@dataclass
class ToolContext:
    """Per-request state shared with the tools.

    Also collects the knowledge-base articles that were actually retrieved, so
    the API can report them back as `sources`.
    """

    customer_id: str | None = None
    sources: list[str] = field(default_factory=list)


class SupportDocsInput(BaseModel):
    """Arguments for `search_support_docs`."""

    query: str = Field(
        description="The customer's question, or the topic to look up, in plain English."
    )


class NoArguments(BaseModel):
    """The (empty) argument schema shared by the account tools.

    Anything the model tries to pass — most often a customer id it made up — is
    logged and dropped, so a lookup can never be redirected to another account.
    Tolerating the stray argument rather than erroring also keeps a small model
    slip from derailing the whole turn.
    """

    model_config = ConfigDict(extra="ignore")

    @model_validator(mode="before")
    @classmethod
    def _drop_unexpected_arguments(cls, data: Any) -> Any:
        if isinstance(data, dict) and data:
            logger.warning(
                "Ignoring unexpected account-tool arguments: %s", sorted(data.keys())
            )
        return data


def _error_payload(code: str, message: str) -> str:
    """Render a tool-level error the model can reason about."""
    return json.dumps({"error": code, "message": message})


def _no_customer_error() -> str:
    """Standard response when an account tool is used with nobody signed in."""
    return _error_payload(
        "not_signed_in",
        "No customer is signed in for this conversation, so account data cannot be "
        "retrieved. Ask the user to sign in.",
    )


def _map_backend_error(exc: Exception, tool_name: str) -> str:
    """Translate an exception from the API client into a tool error payload."""
    if isinstance(exc, CustomerNotFoundError):
        logger.info("%s: customer not found", tool_name)
        return _error_payload(
            "customer_not_found",
            "No account exists for the signed-in customer id. Tell the user their "
            "account could not be found and offer to connect them to a human agent.",
        )
    if isinstance(exc, BackendUnavailableError):
        logger.warning("%s: backend unavailable (%s)", tool_name, exc)
        return _error_payload(
            "backend_unavailable",
            "The account service did not respond. Tell the user their account "
            "information is temporarily unavailable and to try again shortly.",
        )
    if isinstance(exc, MalformedBackendResponseError):
        logger.error("%s: malformed backend response (%s)", tool_name, exc)
        return _error_payload(
            "invalid_response",
            "The account service returned unreadable data. Tell the user their "
            "account information is temporarily unavailable.",
        )
    # Anything else is a bug on our side: log it, but never surface the detail.
    logger.exception("%s: unexpected failure", tool_name)
    return _error_payload(
        "internal_error",
        "The account lookup failed unexpectedly. Tell the user the information is "
        "temporarily unavailable.",
    )


def build_tools(
    context: ToolContext,
    api_client: CustomerAPIClient | None = None,
) -> list[BaseTool]:
    """Build the tool set for one request, scoped to `context.customer_id`.

    Args:
        context: Per-request state; supplies the signed-in customer id and
            collects retrieved knowledge-base sources.
        api_client: Client for the internal customer backend. A default one is
            created when omitted.

    Returns:
        The tools to bind to the model, in the order they are advertised.
    """
    client = api_client or CustomerAPIClient()

    async def search_support_docs(query: str) -> str:
        """Search Northwind Cloud's support documentation.

        Use this for every general question about policies, pricing, plans,
        upgrades, cancellation, refunds, billing rules, password resets, data
        export or contacting support. Returns the most relevant articles.
        """
        try:
            results = get_knowledge_base().search(query, top_k=3)
        except Exception:  # noqa: BLE001 - retrieval must never break a turn
            logger.exception("Knowledge-base search failed for query=%r", query)
            return _error_payload(
                "search_failed",
                "Documentation search is temporarily unavailable. Tell the user you "
                "cannot look up the policy right now.",
            )

        if not results:
            return json.dumps(
                {
                    "results": [],
                    "note": "No matching article. Say the documentation does not cover "
                    "this and offer to open a ticket with a human agent.",
                }
            )

        for result in results:
            if result.document.id not in context.sources:
                context.sources.append(result.document.id)

        return json.dumps(
            {
                "results": [
                    {
                        "id": r.document.id,
                        "title": r.document.title,
                        "content": r.document.content,
                        "relevance": r.score,
                    }
                    for r in results
                ]
            }
        )

    async def _fetch(tool_name: str, client_method: str) -> str:
        """Shared body for the account tools: call the backend, never raise."""
        if not context.customer_id:
            return _no_customer_error()
        try:
            record: Any = await getattr(client, client_method)(context.customer_id)
            return json.dumps(record.model_dump())
        except SupportAgentError as exc:
            return _map_backend_error(exc, tool_name)
        except Exception as exc:  # noqa: BLE001 - see module docstring
            return _map_backend_error(exc, tool_name)

    async def get_customer_account() -> str:
        """Get the signed-in customer's full account record.

        Returns their name, email address, plan, seats, account status,
        renewal date and outstanding balance. Use this for questions like
        "what email is on my account" or "is my account active". Always
        operates on the signed-in customer and takes no arguments.
        """
        return await _fetch("get_customer_account", "get_customer")

    async def get_subscription_details() -> str:
        """Get the signed-in customer's subscription details.

        Returns the plan name, monthly price, currency, number of seats, the
        renewal date and the account status. Use this for "what plan am I on",
        "how much am I paying" and "when does my subscription renew". Always
        operates on the signed-in customer and takes no arguments.
        """
        return await _fetch("get_subscription_details", "get_subscription")

    async def get_billing_information() -> str:
        """Get the signed-in customer's billing information.

        Returns the outstanding balance, currency, payment method, account
        status and recent invoices. Use this for "what do I owe", "what is my
        balance" and invoice questions. Always operates on the signed-in
        customer and takes no arguments.
        """
        return await _fetch("get_billing_information", "get_billing")

    docs_tool = StructuredTool.from_function(
        coroutine=search_support_docs,
        name="search_support_docs",
        description=search_support_docs.__doc__ or "",
        args_schema=SupportDocsInput,
    )
    account_tools: list[BaseTool] = [
        StructuredTool.from_function(
            coroutine=fn,
            name=fn.__name__,
            description=fn.__doc__ or "",
            args_schema=NoArguments,
        )
        for fn in (get_customer_account, get_subscription_details, get_billing_information)
    ]
    return [docs_tool, *account_tools]
