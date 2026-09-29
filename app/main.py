from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv

from app.triage import run_triage
from app.llm_agent import narrate_risk_explanation
from app.agent import PENDING_ACTIONS, approve_action, reject_action, investigate_ticket, build_copilot_agent
from app.conversation import ChatSession
from google.genai import types

load_dotenv()

app = FastAPI(
    title="ShipMate AI - Hybrid LLM + ML Logistics Engine",
    description="Conversational, agentic and explainable triage: Gemini tool-calling agents, "
                "Scikit-Learn delay scoring with Shapley explanations, and human-approved actions",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TICKETS_STORE = []
CHAT_SESSIONS: dict[str, ChatSession] = {}
COPILOT_HISTORY: list = []

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

class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = None
    customer_name: str = "Guest"
    customer_email: str = "guest@example.com"

class CopilotQuestion(BaseModel):
    question: str
    reset: bool = False

def _find_ticket(ticket_id: str) -> dict:
    for ticket in TICKETS_STORE:
        if ticket["ticket_id"] == ticket_id:
            return ticket
    raise HTTPException(status_code=404, detail="Ticket not found")

@app.post("/api/customer/submit")
def customer_submit_ticket(submission: CustomerSubmission):
    ticket_record = run_triage(
        sender_name=submission.sender_name,
        sender_email=submission.sender_email,
        subject=submission.subject,
        body=submission.body,
        tracking_number=submission.tracking_number,
    )
    TICKETS_STORE.insert(0, ticket_record)
    return {"status": "success", "ticket_id": ticket_record["ticket_id"], "ml_prediction": ticket_record["ml_prediction"]}

@app.get("/api/agent/tickets")
def get_all_tickets():
    return TICKETS_STORE

@app.post("/api/agent/update-ticket")
def update_ticket_status(payload: TicketActionPayload):
    ticket = _find_ticket(payload.ticket_id)
    ticket["status"] = payload.status
    if payload.final_reply:
        ticket["final_reply"] = payload.final_reply
    return {"status": "success", "ticket": ticket}

# --- Explainable AI ---

@app.get("/api/agent/tickets/{ticket_id}/explanation")
def get_explanation(ticket_id: str, narrate: bool = False):
    ticket = _find_ticket(ticket_id)
    if narrate and "ml_narrative" not in ticket:
        ticket["ml_narrative"] = narrate_risk_explanation(ticket["ml_prediction"], ticket["ml_explanation"])
    return {
        "ml_prediction": ticket["ml_prediction"],
        "ml_explanation": ticket["ml_explanation"],
        "ml_narrative": ticket.get("ml_narrative"),
        "llm_reasoning": ticket["analysis"].get("urgency_reasoning"),
        "llm_evidence": ticket["analysis"].get("evidence", []),
        "escalation_reasons": ticket.get("escalation_reasons", []),
    }

# --- Agentic AI ---

@app.post("/api/agent/tickets/{ticket_id}/investigate")
def investigate(ticket_id: str):
    ticket = _find_ticket(ticket_id)
    result = investigate_ticket(ticket)
    ticket["agent_summary"] = result["reply"]
    ticket.setdefault("agent_trace", []).extend(result["trace"])
    ticket.setdefault("pending_actions", []).extend(result["new_actions"])
    return result

@app.get("/api/actions/pending")
def list_pending_actions():
    return [a for a in PENDING_ACTIONS.values() if a["status"] == "PENDING_APPROVAL"]

@app.post("/api/actions/{action_id}/approve")
def approve(action_id: str):
    if action_id not in PENDING_ACTIONS:
        raise HTTPException(status_code=404, detail="Action not found")
    return approve_action(action_id)

@app.post("/api/actions/{action_id}/reject")
def reject(action_id: str):
    if action_id not in PENDING_ACTIONS:
        raise HTTPException(status_code=404, detail="Action not found")
    return reject_action(action_id)

# --- Conversational AI ---

@app.post("/api/chat")
def chat(msg: ChatMessage):
    session = CHAT_SESSIONS.get(msg.session_id) if msg.session_id else None
    if session is None:
        session = ChatSession(msg.customer_name, msg.customer_email,
                              on_ticket_created=lambda t: TICKETS_STORE.insert(0, t))
        CHAT_SESSIONS[session.session_id] = session
    turn = session.send(msg.message)
    return {
        "session_id": session.session_id,
        "reply": turn["reply"],
        "trace": turn["trace"],
        "pending_actions": session.pending_actions,
        "handed_off": session.handoff_summary is not None,
        "ticket_id": session.ticket["ticket_id"] if session.ticket else None,
    }

@app.post("/api/copilot")
def copilot(q: CopilotQuestion):
    if q.reset:
        COPILOT_HISTORY.clear()
    COPILOT_HISTORY.append(types.Content(role="user", parts=[types.Part(text=q.question)]))
    return build_copilot_agent(lambda: TICKETS_STORE).run(COPILOT_HISTORY)
