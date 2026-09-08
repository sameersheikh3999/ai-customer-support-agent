"""Internal exception types.

These are deliberately coarse: each one maps to a *user-safe* message, so the
agent and the API layer can explain what went wrong without ever leaking a
stack trace, an upstream URL or a provider error string.
"""

from __future__ import annotations


class SupportAgentError(Exception):
    """Base class for all errors raised inside the application."""

    user_message = "Something went wrong on our side. Please try again shortly."


class BackendUnavailableError(SupportAgentError):
    """The internal customer API timed out, refused the connection or 5xx'd."""

    user_message = (
        "Account information is temporarily unavailable. Please try again in a few minutes."
    )


class CustomerNotFoundError(SupportAgentError):
    """The requested customer id does not exist in the customer backend."""

    user_message = "I could not find an account with that customer ID."


class MalformedBackendResponseError(SupportAgentError):
    """The backend replied, but the payload was not valid/expected JSON."""

    user_message = (
        "Account information is temporarily unavailable. Please try again in a few minutes."
    )


class LLMUnavailableError(SupportAgentError):
    """The OpenAI API failed, timed out, or no API key is configured."""

    user_message = (
        "The assistant is temporarily unavailable. Please try again in a moment, "
        "or email support if it persists."
    )
