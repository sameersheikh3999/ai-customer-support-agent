"""Tests for the mock customer backend and the typed client in front of it."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api_client import CustomerAPIClient
from app.errors import CustomerNotFoundError


def test_health_reports_loaded_data(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] in {"ok", "degraded"}
    assert body["knowledge_base_documents"] == 8
    assert body["customers_loaded"] == 4
    assert "version" in body


def test_customer_roster_lists_demo_accounts(client: TestClient) -> None:
    body = client.get("/api/customers").json()
    ids = {c["customer_id"] for c in body["customers"]}
    assert {"cust_001", "cust_002", "cust_003"} <= ids


def test_get_existing_customer_returns_full_record(client: TestClient) -> None:
    body = client.get("/api/customers/cust_001").json()
    assert body["customer_id"] == "cust_001"
    assert body["name"] == "Sarah Khan"
    assert body["subscription"] == "Pro"
    assert body["monthly_price"] == 49
    assert body["account_status"] == "active"


def test_get_missing_customer_returns_404_without_internals(client: TestClient) -> None:
    response = client.get("/api/customers/cust_999")
    assert response.status_code == 404

    body = response.json()
    assert "cust_999" in body["detail"]
    assert "Traceback" not in body["detail"]


def test_subscription_endpoint_projects_plan_fields(client: TestClient) -> None:
    body = client.get("/api/customers/cust_003/subscription").json()
    assert body == {
        "customer_id": "cust_003",
        "subscription": "Enterprise",
        "monthly_price": 249,
        "currency": "USD",
        "seats": 120,
        "renewal_date": "2027-01-01",
        "account_status": "active",
    }


def test_billing_endpoint_includes_balance_and_invoices(client: TestClient) -> None:
    body = client.get("/api/customers/cust_002/billing").json()
    assert body["outstanding_balance"] == 19
    assert body["account_status"] == "past_due"
    assert body["recent_invoices"][0]["status"] == "unpaid"


async def test_client_returns_validated_models(api_client: CustomerAPIClient) -> None:
    customer = await api_client.get_customer("cust_001")
    assert customer.email == "sarah.khan@example.com"

    subscription = await api_client.get_subscription("cust_001")
    assert subscription.subscription == "Pro"

    billing = await api_client.get_billing("cust_001")
    assert billing.outstanding_balance == 0


async def test_client_raises_typed_error_for_unknown_customer(
    api_client: CustomerAPIClient,
) -> None:
    with pytest.raises(CustomerNotFoundError):
        await api_client.get_customer("cust_404")
