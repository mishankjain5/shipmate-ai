"""
Side-effecting operations. Agents never call these directly: they queue a pending
action, and these run only after a human approves it (see app/agent.py).
"""
import uuid

MAX_VOUCHER_EUR = 200
MANAGER_APPROVAL_ABOVE_EUR = 50


def issue_voucher(customer_email: str, amount_eur: float, reason: str) -> dict:
    code = f"SM-{uuid.uuid4().hex[:8].upper()}"
    print(f"\n🎟️ [VOUCHER] {code} for EUR {amount_eur} to {customer_email} ({reason})\n")
    return {"voucher_code": code, "amount_eur": amount_eur, "sent_to": customer_email}


def request_customs_documents(customer_email: str, tracking_number: str, documents: list) -> dict:
    request_id = f"DOC-{uuid.uuid4().hex[:6].upper()}"
    print(f"\n📄 [CUSTOMS DOC REQUEST] {request_id} for {tracking_number} -> {customer_email}: {documents}\n")
    return {"request_id": request_id, "documents": documents, "upload_link_sent_to": customer_email}
