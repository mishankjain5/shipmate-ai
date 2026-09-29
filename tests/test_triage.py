import pytest
from app.mock_carrier_api import lookup_shipment
from app.ml_delay_model import predict_delay_risk
from app.schemas import TicketPayload, UrgencyLevel

def test_valid_preset_carrier_lookup():
    """Verify that preset tracking IDs return deterministic telemetry."""
    result = lookup_shipment("SHIP-1001")
    assert result is not None
    assert result["status"] == "IN_TRANSIT"
    assert result["carrier"] == "DHL Germany"
    assert result["last_hub"] == "Potsdam Sorting Center"

def test_dynamic_fallback_carrier_lookup():
    """Verify that unlisted tracking codes generate a valid synthetic record."""
    result = lookup_shipment("CUSTOM-9999")
    assert result is not None
    assert result["tracking_number"] == "CUSTOM-9999"
    assert "status" in result
    assert "carrier" in result

def test_none_carrier_lookup():
    """Verify that passing None returns the graceful NO_TRACKING_PROVIDED dict."""
    result = lookup_shipment(None)
    assert result is not None
    assert result["status"] == "NO_TRACKING_PROVIDED"
    assert result["carrier"] == "N/A"

def test_ml_delay_risk_prediction():
    """Verify the Scikit-Learn Random Forest inference output format and bounds."""
    pred = predict_delay_risk(
        carrier="DHL Express",
        last_hub="Roissy CDG Airport Customs",
        dwell_time_hours=48.0,
        is_cross_border=1
    )
    assert "delay_probability" in pred
    assert 0.0 <= pred["delay_probability"] <= 100.0
    assert pred["risk_tier"] in ["LOW_RISK", "MODERATE_RISK", "CRITICAL_RISK"]
    assert "scikit-learn" in pred["model_used"]

def test_ticket_payload_schema():
    """Verify Pydantic input schema initialization."""
    payload = TicketPayload(
        ticket_id="TICK-TEST01",
        sender_email="test@shipmate.de",
        subject="Test delay subject",
        body="Where is my package SHIP-1001?"
    )
    assert payload.ticket_id == "TICK-TEST01"
    assert payload.sender_email == "test@shipmate.de"