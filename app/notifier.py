import httpx
import os

SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")

def send_slack_alert(ticket_id: str, summary: str, urgency: str, carrier_status: dict) -> bool:
    """Sends a rich alert to the operations Slack channel."""
    if not SLACK_WEBHOOK_URL:
        # Fallback to local console log for demo/testing
        print(f"\n📢 [SLACK ALERT - {urgency.upper()}] Ticket {ticket_id}: {summary} | Carrier: {carrier_status}\n")
        return True

    payload = {
        "text": f"🚨 *Urgent Logistics Escalation* | Ticket: `{ticket_id}`",
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"🚨 *Escalated Ticket `{ticket_id}`*\n*Urgency:* {urgency}\n*Summary:* {summary}\n*Tracking Info:* `{carrier_status}`"
                }
            }
        ]
    }
    try:
        response = httpx.post(SLACK_WEBHOOK_URL, json=payload, timeout=5.0)
        return response.status_code == 200
    except Exception as e:
        print(f"Failed to post to Slack: {e}")
        return False