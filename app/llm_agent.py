import os
import re
import time
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from app.schemas import TicketPayload, ExtractedTicketData

load_dotenv()

# Use the latest active Flash model
MODEL_ID = "gemini-3.5-flash-lite"

_client = None

def get_client() -> genai.Client:
    """Lazily creates the Gemini client so modules can be imported (and tested) without an API key."""
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    return _client

MAX_RATE_LIMIT_RETRIES = 2
MAX_RETRY_WAIT_SECONDS = 40

class LLMUnavailableError(RuntimeError):
    """Raised when Gemini stays rate-limited or unavailable after retries."""

def generate_content(client=None, **kwargs):
    """generate_content with retry on 429 (rate limit), honouring the server's suggested retry delay."""
    client = client or get_client()
    for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
        try:
            return client.models.generate_content(**kwargs)
        except errors.APIError as exc:
            if exc.code != 429 or attempt == MAX_RATE_LIMIT_RETRIES:
                raise LLMUnavailableError(f"Gemini API error {exc.code}: {exc.message}") from exc
            match = re.search(r"retry in ([\d.]+)s", str(exc))
            wait = float(match.group(1)) if match else 10.0
            time.sleep(min(wait + 1, MAX_RETRY_WAIT_SECONDS))

SYSTEM_PROMPT = """
You are an expert logistics operations AI triage agent at ShipMate AI.
Your job is to read unstructured customer support tickets, classify them accurately,
extract critical shipping identifiers (Tracking numbers, Customer IDs), and assess urgency.

Rules:
1. If the user mentions damage or threats of legal/dispute actions, mark urgency as CRITICAL or HIGH.
2. Tracking numbers usually follow patterns like 'SEVEN-XXXX' or standard alphanumeric strings (8-16 chars).
3. If no tracking number is found, return null/None.
4. Explain your urgency decision in urgency_reasoning.
5. In evidence, copy 1-3 short phrases EXACTLY as written in the ticket (verbatim, no paraphrasing)
   that justify the category and urgency.
6. Detect the language the customer wrote in and return its ISO 639-1 code.
"""

def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()

def keep_verbatim_quotes(quotes: list[str], source_text: str) -> list[str]:
    """Hallucination guard: drops any 'evidence' quote that does not literally appear in the ticket."""
    haystack = _normalise(source_text)
    return [q for q in quotes if q.strip() and _normalise(q) in haystack]

def analyze_ticket(ticket: TicketPayload) -> ExtractedTicketData:
    prompt = f"Subject: {ticket.subject}\nBody: {ticket.body}\nSender: {ticket.sender_email}"

    response = generate_content(
        model=MODEL_ID,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=ExtractedTicketData,
            temperature=0.0
        ),
    )
    analysis = ExtractedTicketData.model_validate_json(response.text)
    analysis.evidence = keep_verbatim_quotes(analysis.evidence, f"{ticket.subject}\n{ticket.body}")
    return analysis

def generate_draft_response(ticket: TicketPayload, analysis: ExtractedTicketData, carrier_data: dict) -> str:
    draft_prompt = f"""
    Draft a polite, professional, and concise customer support email reply.
    Customer Name/Email: {ticket.sender_email}
    Original Issue: {ticket.body}
    Carrier Verification Details: {carrier_data}
    Action Plan: {analysis.action_required}

    Write the reply in the customer's language (ISO code: {analysis.language}).
    Keep the email under 120 words. No robotic placeholders like '[Insert Date]'.
    """

    response = generate_content(
        model=MODEL_ID,
        contents=draft_prompt,
        config=types.GenerateContentConfig(
            system_instruction="You are a helpful logistics support specialist.",
            temperature=0.3
        ),
    )
    return response.text

def narrate_risk_explanation(ml_prediction: dict, explanation: dict) -> str:
    """Turns Shapley values + counterfactuals into two plain-English sentences for a support agent."""
    prompt = f"""
    A delay-risk model predicted {ml_prediction['delay_probability']}% ({ml_prediction['risk_tier']}).
    Average risk across all shipments is {explanation['base_value']}%.
    Feature contributions in percentage points (Shapley values): {explanation['contributions']}
    Counterfactuals: {explanation['counterfactuals']}
    Data-quality warnings: {explanation['warnings']}

    Explain to a non-technical support agent, in at most 3 sentences, WHY the risk is at this level
    and what would change it. Use only the numbers given above. Mention any data-quality warning briefly.
    """
    response = generate_content(
        model=MODEL_ID,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction="You explain machine-learning predictions clearly and honestly, without jargon.",
            temperature=0.2
        ),
    )
    return response.text
