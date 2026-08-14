from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
import os
import uuid
from datetime import datetime

from app.schemas import TicketPayload, TriageResponse, UrgencyLevel
from app.llm_agent import analyze_ticket, generate_draft_response
from app.mock_carrier_api import lookup_shipment
from app.notifier import send_slack_alert
from app.ml_delay_model import predict_delay_risk  # <--- Import ML module

load_dotenv()

app = FastAPI(
    title="ShipMate AI - Hybrid LLM + ML Logistics Engine",
    description="Dual Portal Hub combining Scikit-Learn Predictive Delay Scoring and LLM Entity Extraction",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TICKETS_STORE = []

class CustomerSubmission(BaseModel):
    sender_name: str
    sender_email: str
    subject: str
    body: str
    tracking_number: Optional[str] = None

class TicketActionPayload(BaseModel):
    ticket_id: str
    final_reply: Optional[str] = None
    status: str

@app.post("/api/customer/submit")
def customer_submit_ticket(submission: CustomerSubmission):
    ticket_id = f"TICK-{uuid.uuid4().hex[:6].upper()}"
    
    payload = TicketPayload(
        ticket_id=ticket_id,
        sender_email=submission.sender_email,
        subject=submission.subject,
        body=submission.body
    )
    
    # 1. LLM Structured NLP Analysis
    analysis = analyze_ticket(payload)
    tracking_to_check = submission.tracking_number or analysis.tracking_number
    if tracking_to_check:
        analysis.tracking_number = tracking_to_check

    # 2. Carrier Telemetry Lookup
    carrier_info = lookup_shipment(analysis.tracking_number)
    
    # 3. Classical ML Tabular Inference (Predicting Delay Likelihood)
    carrier_name = carrier_info.get("carrier", "DHL Express") if carrier_info else "DHL Express"
    hub_name = carrier_info.get("last_hub", "Potsdam Sorting Facility") if carrier_info else "Potsdam Sorting Facility"
    
    # Simulate realistic dwell time based on whether tracking is stalled
    simulated_dwell_time = 44.0 if (carrier_info and carrier_info.get("status") in ["IN_TRANSIT", "CUSTOMS_HOLD"]) else 12.0
    
    ml_prediction = predict_delay_risk(
        carrier=carrier_name,
        last_hub=hub_name,
        dwell_time_hours=simulated_dwell_time,
        is_cross_border=1
    )

    # 4. Decision Engine: Escalation Logic combining LLM + ML + Carrier Exception
    is_high_ml_risk = ml_prediction["risk_tier"] == "CRITICAL_RISK"
    is_customs_hold = carrier_info and carrier_info.get("status") == "CUSTOMS_HOLD"
    is_urgent_tone = analysis.urgency in [UrgencyLevel.HIGH, UrgencyLevel.CRITICAL]

    if is_high_ml_risk or is_customs_hold or is_urgent_tone:
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

    ticket_record = {
        "ticket_id": ticket_id,
        "created_at": datetime.now().strftime("%H:%M:%S"),
        "customer_name": submission.sender_name,
        "customer_email": submission.sender_email,
        "subject": submission.subject,
        "body": submission.body,
        "analysis": analysis.model_dump(),
        "carrier_status": carrier_info,
        "ml_prediction": ml_prediction,  # <--- ML output included in response
        "action_taken": action_taken,
        "draft_reply": draft,
        "status": "OPEN"
    }
    
    TICKETS_STORE.insert(0, ticket_record)
    return {"status": "success", "ticket_id": ticket_id, "ml_prediction": ml_prediction}

@app.get("/api/agent/tickets")
def get_all_tickets():
    return TICKETS_STORE

@app.post("/api/agent/update-ticket")
def update_ticket_status(payload: TicketActionPayload):
    for ticket in TICKETS_STORE:
        if ticket["ticket_id"] == payload.ticket_id:
            ticket["status"] = payload.status
            if payload.final_reply:
                ticket["final_reply"] = payload.final_reply
            return {"status": "success", "ticket": ticket}
    raise HTTPException(status_code=404, detail="Ticket not found")