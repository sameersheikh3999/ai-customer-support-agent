"""Async HTTP client for the internal customer backend.

Every failure mode of a real REST integration is translated here into one of
the application's own exceptions, so callers never have to reason about httpx,
status codes or JSON parsing — and so no upstream detail can leak to the user.
"""

from __future__ import annotations

import logging
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from app.config import get_settings
from app.errors import (
    BackendUnavailableError,
    CustomerNotFoundError,
    MalformedBackendResponseError,
)
from app.schemas import BillingInformation, Customer, SubscriptionDetails

logger = logging.getLogger(__name__)

ModelT = TypeVar("ModelT", bound=BaseModel)


class CustomerAPIClient:
    """Typed wrapper around the customer backend's REST endpoints.

    Args:
        base_url: Root URL of the backend service.
        timeout: Per-request timeout in seconds.
        transport: Optional httpx transport. Tests inject an `ASGITransport`
            so the client talks to the app in-process instead of over TCP.
    """

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        settings = get_settings()
        self.base_url = (base_url or settings.backend_base_url).rstrip("/")
        self.timeout = timeout if timeout is not None else settings.backend_timeout_seconds
        self._transport = transport

    async def _get_json(self, path: str) -> Any:
        """GET `path` and return decoded JSON, mapping every failure to our errors."""
        url = f"{self.base_url}{path}"
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout, transport=self._transport
            ) as client:
                response = await client.get(url)
        except httpx.TimeoutException as exc:
            logger.warning("Customer backend timed out after %ss: %s", self.timeout, exc)
            raise BackendUnavailableError("backend timeout") from exc
        except httpx.HTTPError as exc:
            logger.warning("Customer backend request failed: %s", exc)
            raise BackendUnavailableError("backend transport error") from exc

        if response.status_code == 404:
            raise CustomerNotFoundError(f"404 from backend for {path}")
        if response.status_code >= 500:
            logger.error("Customer backend returned %s for %s", response.status_code, path)
            raise BackendUnavailableError(f"backend {response.status_code}")
        if response.status_code >= 400:
            logger.error("Customer backend rejected %s with %s", path, response.status_code)
            raise MalformedBackendResponseError(f"backend {response.status_code}")

        try:
            return response.json()
        except ValueError as exc:
            logger.error("Customer backend returned non-JSON body for %s", path)
            raise MalformedBackendResponseError("invalid JSON body") from exc

    async def _get_model(self, path: str, model: type[ModelT]) -> ModelT:
        """GET `path` and validate the payload against `model`."""
        payload = await self._get_json(path)
        try:
            return model.model_validate(payload)
        except ValidationError as exc:
            logger.error("Customer backend payload failed validation for %s: %s", path, exc)
            raise MalformedBackendResponseError("unexpected payload shape") from exc

    async def get_customer(self, customer_id: str) -> Customer:
        """Fetch the full account record for `customer_id`."""
        return await self._get_model(f"/api/customers/{customer_id}", Customer)

    async def get_subscription(self, customer_id: str) -> SubscriptionDetails:
        """Fetch subscription/plan details for `customer_id`."""
        return await self._get_model(
            f"/api/customers/{customer_id}/subscription", SubscriptionDetails
        )

    async def get_billing(self, customer_id: str) -> BillingInformation:
        """Fetch billing details and recent invoices for `customer_id`."""
        return await self._get_model(
            f"/api/customers/{customer_id}/billing", BillingInformation
        )
