"""
Conversational layer: a multi-turn customer chat backed by the tool-using agent.

The session keeps the full Gemini history (memory across turns). When the agent hands off
to a human or queues an action that needs approval, the conversation is converted into a
regular triaged ticket so it appears in the agent workspace with its transcript and trace.
"""
import uuid
from typing import Callable, Optional

from google.genai import types

from app.agent import build_customer_agent
from app.triage import run_triage


class ChatSession:
    def __init__(self, customer_name: str, customer_email: str,
                 on_ticket_created: Optional[Callable[[dict], None]] = None, client=None):
        self.session_id = f"CHAT-{uuid.uuid4().hex[:6].upper()}"
        self.customer_name = customer_name
        self.customer_email = customer_email
        self.contents: list = []
        self.transcript: list[dict] = []     # [{"role": "customer"|"assistant", "text", "trace"}]
        self.trace: list[dict] = []
        self.pending_actions: list[dict] = []
        self.handoff_summary: Optional[str] = None
        self.ticket: Optional[dict] = None
        self._on_ticket_created = on_ticket_created
        self._agent = build_customer_agent(customer_name, customer_email, on_handoff=self._handoff, client=client)

    def _handoff(self, summary: str) -> dict:
        self.handoff_summary = summary
        return {"status": "handed_off", "note": "A human agent will join shortly. Tell the customer this."}

    def send(self, text: str) -> dict:
        checkpoint = (len(self.transcript), len(self.contents))
        self.transcript.append({"role": "customer", "text": text})
        self.contents.append(types.Content(role="user", parts=[types.Part(text=text)]))

        try:
            turn = self._agent.run(self.contents)
        except Exception:
            # Roll back so a retry doesn't duplicate the message or leave dangling tool calls
            del self.transcript[checkpoint[0]:]
            del self.contents[checkpoint[1]:]
            raise
        self.trace.extend(turn["trace"])
        self.pending_actions.extend(turn["new_actions"])
        self.transcript.append({"role": "assistant", "text": turn["reply"], "trace": turn["trace"]})

        if self.ticket is None and (self.handoff_summary or self.pending_actions):
            self._create_ticket()
        elif self.ticket is not None:
            self.ticket["handoff_summary"] = self.handoff_summary
        return turn

    def _create_ticket(self) -> None:
        customer_text = "\n".join(m["text"] for m in self.transcript if m["role"] == "customer")
        first_line = self.transcript[0]["text"][:60]
        ticket = run_triage(
            sender_name=self.customer_name,
            sender_email=self.customer_email,
            subject=f"[Chat] {first_line}",
            body=customer_text,
        )
        # Share the live lists so later turns/approvals show up on the ticket automatically
        ticket.update({
            "source": "chat",
            "chat_session_id": self.session_id,
            "handoff_summary": self.handoff_summary,
            "transcript": self.transcript,
            "agent_trace": self.trace,
            "pending_actions": self.pending_actions,
        })
        self.ticket = ticket
        if self._on_ticket_created:
            self._on_ticket_created(ticket)
