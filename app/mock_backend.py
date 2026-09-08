"""A stand-in for the company's internal customer/billing service.

In a real deployment this would be a separate system behind service-to-service
auth. Here it lives in the same FastAPI app but is reached over HTTP by
`app.api_client`, so the agent genuinely performs REST integration rather than
reading a dict in memory.

`BACKEND_FAILURE_MODE` lets you make this service misbehave on demand so the
agent's error handling can be demonstrated (and tested) end to end.
"""

from __future__ import annotations

import asyncio
import json
import logging
from functools import lru_cache
from typing import Any

from fastapi import APIRouter, HTTPException, Response, status

from app.config import DATA_DIR, get_settings
from app.schemas import BillingInformation, Customer, SubscriptionDetails

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["customer-backend"])


@lru_cache(maxsize=1)
def load_customers() -> dict[str, Customer]:
    """Load and validate the fictional customer records from disk."""
    path = DATA_DIR / "customers.json"
    try:
        raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.exception("Could not load customers from %s", path)
        return {}

    customers: dict[str, Customer] = {}
    for customer_id, record in raw.items():
        try:
            customers[customer_id] = Customer.model_validate(record)
        except Exception:  # noqa: BLE001 - one bad row must not break startup
            logger.exception("Skipping invalid customer record %s", customer_id)
    logger.info("Customer backend loaded %d records", len(customers))
    return customers


async def _apply_failure_mode() -> Response | None:
    """Simulate backend faults when BACKEND_FAILURE_MODE is set.

    Returns a `Response` to short-circuit with, or None to continue normally.
    """
    mode = get_settings().backend_failure_mode
    if mode in ("", "none"):
        return None

    logger.warning("Customer backend simulating failure mode=%s", mode)
    if mode == "timeout":
        # Sleep past the client's timeout so httpx raises ReadTimeout. The
        # client gives up first; this task just finishes into the void.
        await asyncio.sleep(get_settings().backend_timeout_seconds * 3)
        return None
    if mode == "server_error":
        raise HTTPException(status_code=503, detail="Customer service unavailable")
    if mode == "malformed":
        return Response(content='{"customer_id": ', media_type="application/json")

    logger.warning("Unknown BACKEND_FAILURE_MODE=%r; ignoring", mode)
    return None


def _get_customer_or_404(customer_id: str) -> Customer:
    """Look up a customer, raising a 404 when it does not exist."""
    customer = load_customers().get(customer_id)
    if customer is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No customer with id '{customer_id}'",
        )
    return customer


@router.get("/customers", summary="List demo customer ids")
async def list_customers() -> dict[str, list[dict[str, str]]]:
    """Return the demo roster so the UI can populate its customer selector."""
    return {
        "customers": [
            {"customer_id": c.customer_id, "name": c.name, "subscription": c.subscription}
            for c in load_customers().values()
        ]
    }


@router.get(
    "/customers/{customer_id}",
    response_model=Customer,
    summary="Full customer record",
)
async def get_customer(customer_id: str) -> Any:
    """Return the complete account record for a customer."""
    if (short_circuit := await _apply_failure_mode()) is not None:
        return short_circuit
    return _get_customer_or_404(customer_id)


@router.get(
    "/customers/{customer_id}/subscription",
    response_model=SubscriptionDetails,
    summary="Subscription details",
)
async def get_subscription(customer_id: str) -> Any:
    """Return plan, price, seats and renewal date for a customer."""
    if (short_circuit := await _apply_failure_mode()) is not None:
        return short_circuit
    customer = _get_customer_or_404(customer_id)
    return SubscriptionDetails(
        customer_id=customer.customer_id,
        subscription=customer.subscription,
        monthly_price=customer.monthly_price,
        currency=customer.currency,
        seats=customer.seats,
        renewal_date=customer.renewal_date,
        account_status=customer.account_status,
    )


@router.get(
    "/customers/{customer_id}/billing",
    response_model=BillingInformation,
    summary="Billing information",
)
async def get_billing(customer_id: str) -> Any:
    """Return balance, payment method and recent invoices for a customer."""
    if (short_circuit := await _apply_failure_mode()) is not None:
        return short_circuit
    customer = _get_customer_or_404(customer_id)
    return BillingInformation(
        customer_id=customer.customer_id,
        outstanding_balance=customer.outstanding_balance,
        currency=customer.currency,
        payment_method=customer.payment_method,
        account_status=customer.account_status,
        recent_invoices=customer.recent_invoices,
    )
