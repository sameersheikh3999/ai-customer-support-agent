"""Per-session conversation memory.

Deliberately simple: an in-process dict of session id -> recent messages, with
a size cap and a TTL so a long-running demo does not grow without bound. A real
deployment would put this in Redis or Postgres (see README limitations).
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class Session:
    """One conversation thread."""

    session_id: str
    messages: deque[BaseMessage] = field(default_factory=deque)
    last_seen: float = field(default_factory=time.time)


class ConversationMemory:
    """Thread-safe, bounded, in-memory chat history keyed by session id."""

    def __init__(self, max_messages: int | None = None, ttl_seconds: int | None = None) -> None:
        settings = get_settings()
        self.max_messages = max_messages or settings.history_max_messages
        self.ttl_seconds = ttl_seconds or settings.session_ttl_seconds
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def _purge_expired(self, now: float) -> None:
        """Drop sessions untouched for longer than the TTL. Caller holds the lock."""
        expired = [
            sid for sid, s in self._sessions.items() if now - s.last_seen > self.ttl_seconds
        ]
        for session_id in expired:
            del self._sessions[session_id]
        if expired:
            logger.debug("Purged %d expired session(s)", len(expired))

    def get_history(self, session_id: str) -> list[BaseMessage]:
        """Return the recent messages for a session, oldest first."""
        now = time.time()
        with self._lock:
            self._purge_expired(now)
            session = self._sessions.get(session_id)
            if session is None:
                return []
            session.last_seen = now
            return list(session.messages)

    def append_turn(self, session_id: str, user_message: str, assistant_message: str) -> None:
        """Record one complete exchange, trimming to the configured window."""
        now = time.time()
        with self._lock:
            session = self._sessions.setdefault(session_id, Session(session_id=session_id))
            session.messages.append(HumanMessage(content=user_message))
            session.messages.append(AIMessage(content=assistant_message))
            while len(session.messages) > self.max_messages:
                session.messages.popleft()
            session.last_seen = now

    def clear(self, session_id: str) -> bool:
        """Forget a session. Returns True when something was removed."""
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def reset(self) -> None:
        """Drop every session (used by tests)."""
        with self._lock:
            self._sessions.clear()

    @property
    def active_sessions(self) -> int:
        """Number of sessions currently held in memory."""
        with self._lock:
            return len(self._sessions)


# One shared store for the process.
conversation_memory = ConversationMemory()
