from datetime import datetime

import pytest

from app.mock_carrier_api import SimulatedCarrierClient, lookup_shipment
from app.ml_delay_model import CARRIERS, HUBS
from app.triage import assess_delay_risk

FIXED_NOW = datetime(2026, 9, 29, 12, 0)
client = SimulatedCarrierClient(now_fn=lambda: FIXED_NOW)
SAMPLE = [f"SHIP-{n}" for n in range(4000, 4200)]


def test_same_tracking_number_always_returns_same_shipment():
    assert client.lookup("SHIP-4711") == client.lookup("ship-4711 ")


def test_invalid_format_is_not_found():
    for code in ["abc", "!!!!!!!!", "NODIGITSHERE"]:
        assert lookup_shipment(code)["status"] == "NOT_FOUND"


def test_simulated_shipments_use_model_vocabulary():
    for code in SAMPLE:
        s = client.lookup(code)
        if s["status"] != "NOT_FOUND":
            assert s["carrier"] in CARRIERS
            assert s["last_hub"] in HUBS


def test_outcome_mix_is_realistic():
    statuses = [client.lookup(c)["status"] for c in SAMPLE]
    for status in ["DELIVERED", "IN_TRANSIT", "CUSTOMS_HOLD", "NOT_FOUND"]:
        assert status in statuses


def test_scan_history_is_chronological_and_matches_dwell_time():
    for code in SAMPLE:
        s = client.lookup(code)
        if s["status"] in ("IN_TRANSIT", "CUSTOMS_HOLD"):
            times = [datetime.strptime(e["timestamp"], "%Y-%m-%d %H:%M") for e in s["scan_events"]]
            assert times == sorted(times)
            hours_since_last_scan = (FIXED_NOW - times[-1]).total_seconds() / 3600
            assert hours_since_last_scan == pytest.approx(s["dwell_time_hours"], abs=0.1)


def test_customs_hold_has_exception_reason():
    s = lookup_shipment("SHIP-2002")
    assert s["status"] == "CUSTOMS_HOLD" and s["exception_flag"]
    assert s["exception_reason"] == "Missing Commercial Invoice"
    assert s["origin"].endswith("CH")          # non-EU origin, so customs is plausible


@pytest.mark.parametrize("code", ["SHIP-3003", None, "abc"])
def test_ml_is_skipped_when_there_is_no_shipment_in_transit(code):
    prediction, explanation = assess_delay_risk(lookup_shipment(code))
    assert prediction is None and explanation is None


def test_ml_uses_live_dwell_time_and_has_no_vocabulary_warnings():
    prediction, explanation = assess_delay_risk(lookup_shipment("SHIP-1001"))
    assert prediction["risk_tier"] == "CRITICAL_RISK"
    assert explanation["warnings"] == []
    dwell = next(c for c in explanation["contributions"] if c["feature"] == "dwell_time_hours")
    assert dwell["value"] == 70.0
