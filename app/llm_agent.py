import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from app.schemas import TicketPayload, ExtractedTicketData

load_dotenv()

# Initialize Gemini Client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Use the latest active Flash model
MODEL_ID = "gemini-3.5-flash-lite"

SYSTEM_PROMPT = """
You are an expert logistics operations AI triage agent at Seven Senders.
Your job is to read unstructured customer support tickets, classify them accurately, 
extract critical shipping identifiers (Tracking numbers, Customer IDs), and assess urgency.

Rules:
1. If the user mentions damage or threats of legal/dispute actions, mark urgency as CRITICAL or HIGH.
2. Tracking numbers usually follow patterns like 'SEVEN-XXXX' or standard alphanumeric strings (8-16 chars).
3. If no tracking number is found, return null/None.
"""

def analyze_ticket(ticket: TicketPayload) -> ExtractedTicketData:
    prompt = f"Subject: {ticket.subject}\nBody: {ticket.body}\nSender: {ticket.sender_email}"
    
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=ExtractedTicketData,
            temperature=0.0
        ),
    )
    return ExtractedTicketData.model_validate_json(response.text)

def generate_draft_response(ticket: TicketPayload, analysis: ExtractedTicketData, carrier_data: dict) -> str:
    draft_prompt = f"""
    Draft a polite, professional, and concise customer support email reply.
    Customer Name/Email: {ticket.sender_email}
    Original Issue: {ticket.body}
    Carrier Verification Details: {carrier_data}
    Action Plan: {analysis.action_required}
    
    Keep the email under 120 words. No robotic placeholders like '[Insert Date]'.
    """
    
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=draft_prompt,
        config=types.GenerateContentConfig(
            system_instruction="You are a helpful logistics support specialist.",
            temperature=0.3
        ),
    )
    return response.text