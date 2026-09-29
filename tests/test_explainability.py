import pytest
from app.explainability import explain_prediction
from app.ml_delay_model import predict_delay_risk
from app.llm_agent import keep_verbatim_quotes
from app.schemas import ExtractedTicketData, TicketCategory, UrgencyLevel
from app.triage import decide_escalation


@pytest.mark.parametrize("carrier,hub,dwell", [
    ("Colissimo", "Roissy CDG Airport Customs", 44.0),
    ("PostNL", "Amsterdam Parcel Center", 12.0),
    ("DHL Germany", "Potsdam Sorting Center", 44.0),
])
def test_shapley_values_sum_to_prediction(carrier, hub, dwell):
    """Efficiency axiom: base value + sum of contributions == model output."""
    exp = explain_prediction(carrier, hub, dwell, 1)
    total = exp["base_value"] + sum(c["impact_pct_points"] for c in exp["contributions"])
    assert total == pytest.approx(exp["prediction"], abs=0.3)
    assert exp["prediction"] == pytest.approx(predict_delay_risk(carrier, hub, dwell, 1)["delay_probability"], abs=0.1)


def test_dwell_time_is_top_driver_for_stalled_parcel():
    exp = explain_prediction("DHL Express", "Leipzig Hub", 60.0, 1)
    assert exp["contributions"][0]["feature"] == "dwell_time_hours"
    assert exp["contributions"][0]["impact_pct_points"] > 0


def test_unknown_categories_are_flagged():
    exp = explain_prediction("DHL Germany", "Potsdam Sorting Center", 44.0, 1)
    assert len(exp["warnings"]) == 2
    assert explain_prediction("Colissimo", "Roissy CDG Airport Customs", 44.0, 1)["warnings"] == []


def test_counterfactuals_present():
    exp = explain_prediction("PostNL", "Amsterdam Parcel Center", 12.0, 1)
    assert any("CRITICAL" in cf for cf in exp["counterfactuals"])


def test_hallucinated_evidence_is_discarded():
    text = "Our package arrived completely   crushed, torn open."
    quotes = ["arrived completely crushed", "customer is threatening to sue"]
    assert keep_verbatim_quotes(quotes, text) == ["arrived completely crushed"]


def _analysis(urgency):
    return ExtractedTicketData(category=TicketCategory.DELAYED_DELIVERY, urgency=urgency,
                               summary="s", action_required="a", urgency_reasoning="because")


def test_escalation_trace_lists_every_rule_that_fired():
    reasons = decide_escalation(
        _analysis(UrgencyLevel.HIGH),
        {"risk_tier": "CRITICAL_RISK", "delay_probability": 91.0},
        {"status": "CUSTOMS_HOLD", "exception_reason": "Missing Commercial Invoice"},
    )
    assert len(reasons) == 3
    assert "Missing Commercial Invoice" in reasons[1]


def test_no_escalation_when_nothing_fires():
    reasons = decide_escalation(_analysis(UrgencyLevel.LOW),
                                {"risk_tier": "LOW_RISK", "delay_probability": 4.0},
                                {"status": "DELIVERED"})
    assert reasons == []
