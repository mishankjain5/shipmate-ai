"""Mock company knowledge the agent can consult: support policies and CRM customer history."""
from typing import Dict

POLICIES: Dict[str, str] = {
    "damaged_goods": (
        "Damaged goods: customer must report within 7 days of delivery. For damage confirmed by photo or "
        "description, offer replacement or full refund. A goodwill voucher of up to EUR 20 may be offered. "
        "Refunds above EUR 50 require manager approval."
    ),
    "delayed_delivery": (
        "Delayed delivery: parcels are considered late after 48h without a new scan. Standard shipments "
        "delayed more than 3 days qualify for a EUR 5-10 goodwill voucher. Proactively contact the carrier "
        "if dwell time at a hub exceeds 36h."
    ),
    "customs": (
        "Customs holds (EU cross-border): the most common cause is a missing commercial invoice. Required "
        "documents: commercial invoice (with HS codes and item values), proof of export, and recipient "
        "EORI number for business shipments. Clearance usually takes 1-2 business days after documents arrive."
    ),
    "lost_package": (
        "Lost package: a parcel is declared lost after 10 business days without a scan. Open a carrier "
        "investigation first; refund or reship once the carrier confirms loss."
    ),
    "not_received": (
        "Delivered but not received: the carrier shows DELIVERED but the customer does not have the parcel. "
        "1) Ask the customer to check with neighbours, safe places and their mailbox, and confirm the delivery "
        "address. 2) Open a carrier investigation to obtain proof of delivery (signature, photo, GPS location). "
        "3) If the carrier cannot prove delivery within 3 business days, reship or refund. Never tell the "
        "customer the matter is closed just because the tracking says delivered."
    ),
    "address_change": (
        "Address change: possible only while status is IN_TRANSIT and before the final-mile hub. "
        "Delivered or customs-held parcels cannot be rerouted."
    ),
    "vouchers": (
        "Vouchers: agents may issue up to EUR 50 on their own approval. EUR 50-200 needs a manager. "
        "Above EUR 200 is not allowed via vouchers."
    ),
}

CUSTOMER_HISTORY: Dict[str, Dict] = {
    "max.mustermann@post.de": {
        "customer_since": "2023-04",
        "orders_last_12m": 14,
        "previous_tickets": 1,
        "previous_issues": ["delayed_delivery (resolved, EUR 5 voucher)"],
        "segment": "loyal consumer",
    },
    "c.dupont@paris-store.fr": {
        "customer_since": "2021-11",
        "orders_last_12m": 212,
        "previous_tickets": 6,
        "previous_issues": ["customs hold x3 (missing invoice)", "address_change"],
        "segment": "B2B merchant - high value",
    },
    "jan.vries@enterprise.nl": {
        "customer_since": "2022-06",
        "orders_last_12m": 87,
        "previous_tickets": 2,
        "previous_issues": ["damaged_item (refunded)"],
        "segment": "B2B enterprise",
    },
}


def check_policy(topic: str) -> dict:
    """Keyword lookup over the policy handbook (stand-in for a RAG retriever)."""
    key = topic.strip().lower().replace(" ", "_")
    matches = {k: v for k, v in POLICIES.items() if k in key or key in k or any(w in k for w in key.split("_") if len(w) > 3)}
    if not matches:
        return {"found": False, "available_topics": list(POLICIES)}
    return {"found": True, "policies": matches}


def get_customer_history(email: str) -> dict:
    record = CUSTOMER_HISTORY.get(email.strip().lower())
    if not record:
        return {"known_customer": False, "note": "No previous orders or tickets on file."}
    return {"known_customer": True, **record}
