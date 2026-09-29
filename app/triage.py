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


SCORABLE_STATUSES = {"IN_TRANSIT", "CUSTOMS_HOLD"}


def ml_not_applicable_reason(carrier_info: Optional[dict]) -> Optional[str]:
    """Why the delay model should NOT score this shipment (None = it should)."""
    status = (carrier_info or {}).get("status")
    if status in SCORABLE_STATUSES:
        return None
    return {
        "NO_TRACKING_PROVIDED": "No tracking number was provided, so there is no shipment to score.",
        "NOT_FOUND": "The carrier has no record of this tracking number.",
        "DELIVERED": "The parcel has already been delivered, so there is no delay left to predict.",
    }.get(status, f"Shipment status '{status}' cannot be scored.")


def ml_inputs_from_carrier(carrier_info: dict) -> dict:
    """Derives delay-model features from a live carrier record."""
    return {
        "carrier": carrier_info["carrier"],
        "last_hub": carrier_info["last_hub"],
        "dwell_time_hours": carrier_info["dwell_time_hours"],   # hours since the last carrier scan
        "is_cross_border": carrier_info["is_cross_border"],
    }


def assess_delay_risk(carrier_info: Optional[dict]) -> tuple[Optional[dict], Optional[dict]]:
    """Returns (ml_prediction, ml_explanation), or (None, None) if there is no shipment in transit to score."""
    if ml_not_applicable_reason(carrier_info):
        return None, None
    features = ml_inputs_from_carrier(carrier_info)
    prediction = predict_delay_risk(**features)
    explanation = explain_prediction(**features)
    explanation["notes"] = [f"Dwell time is computed from the last carrier scan ({features['dwell_time_hours']}h ago)."]
    return prediction, explanation


def decide_escalation(analysis: ExtractedTicketData, ml_prediction: Optional[dict], carrier_info: Optional[dict]) -> list[str]:
    """Decision engine combining LLM + ML + carrier exception. Returns the reasons that fired (empty = no escalation)."""
    reasons = []
    if ml_prediction and ml_prediction["risk_tier"] == "CRITICAL_RISK":
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
        risk_label = f"{ml_prediction['delay_probability']}%" if ml_prediction else "n/a"
        send_slack_alert(
            ticket_id=ticket_id,
            summary=f"[ML Risk: {risk_label}] {analysis.summary}",
            urgency=analysis.urgency.value,
            carrier_status={k: v for k, v in carrier_info.items() if k != "scan_events"}
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
        "ml_not_applicable": ml_not_applicable_reason(carrier_info),
        "action_taken": action_taken,
        "escalation_reasons": escalation_reasons,
        "draft_reply": draft,
        "pending_actions": [],
        "agent_trace": [],
        "status": "OPEN"
    }
