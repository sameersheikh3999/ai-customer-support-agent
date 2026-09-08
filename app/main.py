"""FastAPI application: chat endpoint, health check, mock backend and UI.

Error handling policy for the whole app:
  * expected failures  -> a typed exception with a user-safe `user_message`
  * unexpected failures -> logged with a traceback, returned as a generic 500
Under no circumstance does a stack trace, upstream URL or provider message
reach the client.
"""

from __future__ import annotations

import logging
import re
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from http import HTTPStatus

from fastapi import Depends, FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from app import __version__
from app.agent import SupportAgent
from app.config import STATIC_DIR, configure_logging, get_settings
from app.errors import (
    BackendUnavailableError,
    CustomerNotFoundError,
    LLMUnavailableError,
    SupportAgentError,
)
from app.knowledge_base import get_knowledge_base
from app.memory import conversation_memory
from app.mock_backend import load_customers
from app.mock_backend import router as customer_backend_router
from app.schemas import ChatRequest, ChatResponse, ErrorResponse, HealthResponse

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Warm the on-disk caches at startup so the first request is not slow."""
    configure_logging()
    settings = get_settings()
    documents = len(get_knowledge_base().documents)
    customers = len(load_customers())
    logger.info(
        "Startup: model=%s llm_configured=%s docs=%d customers=%d backend=%s",
        settings.openai_model,
        settings.llm_configured,
        documents,
        customers,
        settings.backend_base_url,
    )
    if not settings.llm_configured:
        logger.warning("OPENAI_API_KEY is not set — /chat will return 503 until it is.")
    yield


app = FastAPI(
    title="AI Customer Support Agent",
    description=(
        "A LangChain tool-calling support assistant for a fictional SaaS company. "
        "General questions are answered from a retrieved knowledge base; account "
        "questions are answered only from the internal customer API."
    ),
    version=__version__,
    lifespan=lifespan,
)

app.include_router(customer_backend_router)


# --------------------------------------------------------------------------
# Dependencies
# --------------------------------------------------------------------------
def get_agent() -> SupportAgent:
    """Provide the agent for a request.

    Declared as a dependency so tests can override it with a scripted fake
    model and never call the real OpenAI API. The OpenAI client itself is
    built lazily inside the agent, so this never raises during dependency
    resolution — which would otherwise pre-empt request validation.
    """
    return SupportAgent()


# --------------------------------------------------------------------------
# Exception handlers
# --------------------------------------------------------------------------
@app.exception_handler(RequestValidationError)
async def handle_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    """Turn Pydantic validation failures into a short, readable message."""
    problems = "; ".join(
        f"{'.'.join(str(p) for p in err.get('loc', []) if p != 'body')}: {err.get('msg')}"
        for err in exc.errors()
    )
    logger.info("Rejected invalid request: %s", problems)
    return JSONResponse(
        status_code=422,  # literal: the Starlette constant was renamed mid-1.x
        content=ErrorResponse(
            error="invalid_request",
            detail=problems or "The request body was not valid.",
        ).model_dump(),
    )


@app.exception_handler(SupportAgentError)
async def handle_support_agent_error(_: Request, exc: SupportAgentError) -> JSONResponse:
    """Map our own exceptions to a status code and their user-safe message."""
    status_code = {
        CustomerNotFoundError: status.HTTP_404_NOT_FOUND,
        BackendUnavailableError: status.HTTP_503_SERVICE_UNAVAILABLE,
        LLMUnavailableError: status.HTTP_503_SERVICE_UNAVAILABLE,
    }.get(type(exc), status.HTTP_500_INTERNAL_SERVER_ERROR)

    logger.warning("%s: %s", type(exc).__name__, exc)
    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(
            error=_snake_case(type(exc).__name__), detail=exc.user_message
        ).model_dump(),
    )


@app.exception_handler(StarletteHTTPException)
async def handle_http_exception(_: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Give FastAPI's own HTTP errors the same body shape as ours."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=HTTPStatus(exc.status_code).phrase.lower().replace(" ", "_"),
            detail=str(exc.detail),
        ).model_dump(),
        headers=getattr(exc, "headers", None),
    )


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, _: Exception) -> JSONResponse:
    """Last line of defence: log the traceback, return a generic message."""
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="internal_error",
            detail="Something went wrong on our side. Please try again shortly.",
        ).model_dump(),
    )


def _snake_case(name: str) -> str:
    """`BackendUnavailableError` -> `backend_unavailable`; acronyms stay intact."""
    trimmed = name.removesuffix("Error")
    spaced = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", trimmed)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", spaced).lower()


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    """Report liveness plus whether the app's dependencies are configured."""
    settings = get_settings()
    documents = len(get_knowledge_base().documents)
    customers = len(load_customers())
    healthy = settings.llm_configured and documents > 0 and customers > 0
    return HealthResponse(
        status="ok" if healthy else "degraded",
        version=__version__,
        llm_configured=settings.llm_configured,
        knowledge_base_documents=documents,
        customers_loaded=customers,
    )


@app.post(
    "/chat",
    response_model=ChatResponse,
    tags=["chat"],
    responses={
        422: {"model": ErrorResponse, "description": "Invalid request body"},
        503: {"model": ErrorResponse, "description": "A dependency is unavailable"},
    },
)
async def chat(
    payload: ChatRequest, agent: SupportAgent = Depends(get_agent)
) -> ChatResponse:
    """Answer one message, using conversation history from `session_id`."""
    session_id = payload.session_id or f"session_{uuid.uuid4().hex[:12]}"
    history = conversation_memory.get_history(session_id)

    logger.info(
        "chat session=%s customer=%s history=%d chars=%d",
        session_id,
        payload.customer_id or "-",
        len(history),
        len(payload.message),
    )

    result = await agent.answer(
        message=payload.message,
        customer_id=payload.customer_id,
        history=history,
    )

    # Only remember exchanges that actually succeeded, so a transient outage
    # does not poison the thread with "I couldn't reach the service" turns.
    if not result.degraded:
        conversation_memory.append_turn(session_id, payload.message, result.answer)

    return ChatResponse(
        answer=result.answer,
        tools_used=result.tools_used,
        session_id=session_id,
        sources=result.sources,
        degraded=result.degraded,
    )


@app.delete("/sessions/{session_id}", tags=["chat"])
async def clear_session(session_id: str) -> dict[str, bool | str]:
    """Forget a conversation's history."""
    return {"session_id": session_id, "cleared": conversation_memory.clear(session_id)}


# Serve the demo UI last so it does not shadow the API routes above.
if STATIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="ui")
