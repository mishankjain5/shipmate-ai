"""Shared triage pipeline used by both the FastAPI backend and the Streamlit app."""
import uuid
from datetime import datetime
from typing import Optional

from app.schemas import TicketPayload, ExtractedTicketData, UrgencyLevel
from app.llm_agent import analyze_ticket, generate_draft_response
from app.mock_carrier_api import lookup_shipment
from app.ml_delay_model import predict_delay_risk
from app.explainability import explain_prediction
from app.notifier import send_slack_alert


def new_ticket_id() -> str:
    return f"TICK-{uuid.uuid4().hex[:6].upper()}"


def ml_inputs_from_carrier(carrier_info: Optional[dict]) -> dict:
    """Derives delay-model features from a carrier record."""
    carrier_name = carrier_info.get("carrier", "DHL Express") if carrier_info else "DHL Express"
    hub_name = carrier_info.get("last_hub", "Potsdam Sorting Facility") if carrier_info else "Potsdam Sorting Facility"
    # Simulate realistic dwell time based on whether tracking is stalled
    simulated_dwell_time = 44.0 if (carrier_info and carrier_info.get("status") in ["IN_TRANSIT", "CUSTOMS_HOLD"]) else 12.0
    return {
        "carrier": carrier_name,
        "last_hub": hub_name,
        "dwell_time_hours": simulated_dwell_time,
        "is_cross_border": 1,
    }


def assess_delay_risk(carrier_info: Optional[dict]) -> tuple[dict, dict]:
    """Returns (ml_prediction, ml_explanation) for a carrier record."""
    features = ml_inputs_from_carrier(carrier_info)
    prediction = predict_delay_risk(**features)
    explanation = explain_prediction(**features)
    explanation["notes"] = ["dwell_time_hours is simulated from carrier status (44h if stalled, else 12h) in this demo."]
    return prediction, explanation


def decide_escalation(analysis: ExtractedTicketData, ml_prediction: dict, carrier_info: Optional[dict]) -> list[str]:
    """Decision engine combining LLM + ML + carrier exception. Returns the reasons that fired (empty = no escalation)."""
    reasons = []
    if ml_prediction["risk_tier"] == "CRITICAL_RISK":
        reasons.append(f"ML delay risk is CRITICAL ({ml_prediction['delay_probability']}% >= 70%)")
    if carrier_info and carrier_info.get("status") == "CUSTOMS_HOLD":
        detail = carrier_info.get("exception_reason", "no reason given")
        reasons.append(f"Carrier reports CUSTOMS_HOLD ({detail})")
    if analysis.urgency in [UrgencyLevel.HIGH, UrgencyLevel.CRITICAL]:
        reasons.append(f"LLM rated urgency {analysis.urgency.value.upper()}: {analysis.urgency_reasoning or analysis.summary}")
    return reasons


def run_triage(sender_name: str, sender_email: str, subject: str, body: str,
               tracking_number: Optional[str] = None, ticket_id: Optional[str] = None) -> dict:
    ticket_id = ticket_id or new_ticket_id()
    payload = TicketPayload(ticket_id=ticket_id, sender_email=sender_email, subject=subject, body=body)

    # 1. LLM Structured NLP Analysis
    analysis = analyze_ticket(payload)
    tracking_to_check = tracking_number or analysis.tracking_number
    if tracking_to_check:
        analysis.tracking_number = tracking_to_check

    # 2. Carrier Telemetry Lookup
    carrier_info = lookup_shipment(analysis.tracking_number)

    # 3. Classical ML Tabular Inference + Explainability
    ml_prediction, ml_explanation = assess_delay_risk(carrier_info)

    # 4. Decision Engine with an explicit decision trace
    escalation_reasons = decide_escalation(analysis, ml_prediction, carrier_info)
    if escalation_reasons:
        send_slack_alert(
            ticket_id=ticket_id,
            summary=f"[ML Risk: {ml_prediction['delay_probability']}%] {analysis.summary}",
            urgency=analysis.urgency.value,
            carrier_status=carrier_info
        )
        action_taken = "ESCALATED_TO_SLACK"
    else:
        action_taken = "AUTO_DRAFT_CREATED"

    draft = generate_draft_response(payload, analysis, carrier_info or {"note": "No shipment record found"})

    return {
        "ticket_id": ticket_id,
        "created_at": datetime.now().strftime("%H:%M:%S"),
        "source": "form",
        "customer_name": sender_name,
        "customer_email": sender_email,
        "subject": subject,
        "body": body,
        "analysis": analysis.model_dump(),
        "carrier_status": carrier_info,
        "ml_prediction": ml_prediction,
        "ml_explanation": ml_explanation,
        "action_taken": action_taken,
        "escalation_reasons": escalation_reasons,
        "draft_reply": draft,
        "pending_actions": [],
        "agent_trace": [],
        "status": "OPEN"
    }
