import html
import json
import re
from contextlib import contextmanager

import altair as alt
import pandas as pd
import streamlit as st
from google.genai import types

# Internal project modules
from app.triage import run_triage
from app.llm_agent import narrate_risk_explanation, LLMUnavailableError
from app.agent import approve_action, reject_action, investigate_ticket, build_copilot_agent
from app.conversation import ChatSession

# --- Page Configuration ---
st.set_page_config(
    page_title="ShipMate AI | Logistics Triage & ML Risk Suite",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E293B; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1rem; color: #64748B; margin-bottom: 1.5rem; }
    .badge-high { background-color: #FEE2E2; color: #991B1B; padding: 4px 8px; border-radius: 6px; font-weight: 600; font-size: 0.75rem; }
    .badge-open { background-color: #FEF3C7; color: #92400E; padding: 4px 8px; border-radius: 6px; font-weight: 600; font-size: 0.75rem; }
    .badge-resolved { background-color: #D1FAE5; color: #065F46; padding: 4px 8px; border-radius: 6px; font-weight: 600; font-size: 0.75rem; }
    .customer-msg { background-color: #EFF6FF; color: #1E293B; padding: 12px 14px; border-radius: 8px; line-height: 1.5; }
    .customer-msg mark { background-color: #FDE68A; padding: 0 2px; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# --- State Management ---
if "tickets" not in st.session_state:
    st.session_state.tickets = []
if "chat_session" not in st.session_state:
    st.session_state.chat_session = None
if "copilot_contents" not in st.session_state:
    st.session_state.copilot_contents = []
    st.session_state.copilot_log = []

DEMO_SCENARIOS = {
    "Delayed Delivery (Potsdam)": {
        "name": "Max Mustermann",
        "email": "max.mustermann@post.de",
        "subject": "Where is my package? Delivery delayed!",
        "tracking": "SHIP-1001",
        "body": "Hi, I have been tracking order SHIP-1001 for 3 days and the status has not moved from Potsdam. I need this urgently for a birthday party tomorrow or I will cancel the order."
    },
    "Customs Exception (Paris CDG)": {
        "name": "Claire Dupont",
        "email": "c.dupont@paris-store.fr",
        "subject": "Colis bloqué à la douane / Customs Hold",
        "tracking": "SHIP-2002",
        "body": "Hello, my shipment SHIP-2002 to Paris is marked with a customs hold exception at CDG airport. Can you check what documents are missing to clear this?"
    },
    "Damaged Goods (Amsterdam)": {
        "name": "Jan de Vries",
        "email": "jan.vries@enterprise.nl",
        "subject": "DAMAGED GOODS ON ARRIVAL - REFUND NEEDED",
        "tracking": "SHIP-3003",
        "body": "Our package SHIP-3003 arrived completely crushed, torn open, and unusable. Please process an immediate replacement or full refund right away."
    }
}

CHAT_STARTERS = [
    "Hi, my parcel hasn't arrived yet and I'm getting worried.",
    "Mon colis SHIP-2002 est bloqué à la douane. Quels documents faut-il envoyer ?",
    "SHIP-3003 arrived crushed. I want a refund or I'm calling my lawyer.",
]

COPILOT_STARTERS = [
    "Give me an overview of the queue.",
    "Which tickets have a delay risk above 70%?",
    "Why was the most recent ticket escalated?",
]

# --- UI helpers ---
@contextmanager
def llm_errors():
    """Shows a friendly message instead of a traceback when Gemini is rate-limited or down."""
    try:
        yield
    except LLMUnavailableError as exc:
        st.error("⏳ The AI service is busy or unavailable (the Gemini free tier allows 15 requests/minute). "
                 f"Please wait ~30 seconds and try again.\n\n`{exc}`")
        render_sidebar_metrics()
        st.stop()


def render_trace(trace: list, title: str = "Agent reasoning trace"):
    if not trace:
        return
    icons = {"tool": "🔧", "queued": "⏸️", "blocked": "⛔", "error": "⚠️", "answer": "💬", "limit": "🛑"}
    tool_steps = [s for s in trace if s["type"] != "answer"]
    with st.expander(f"🧭 {title} ({len(tool_steps)} tool calls)"):
        for s in trace:
            icon = icons.get(s["type"], "•")
            if s["type"] == "answer":
                st.markdown(f"{icon} **Step {s['step']}** - final answer")
                continue
            args = ", ".join(f"{k}={v!r}" for k, v in s["args"].items())
            label = {"queued": "queued for human approval", "blocked": "blocked by guardrail"}.get(s["type"], "")
            st.markdown(f"{icon} **Step {s['step']}** - `{s['tool']}({args})` {label}")
            st.code(json.dumps(s["result"], indent=1, default=str)[:1200], language="json")


def render_pending_actions(actions: list, key_prefix: str):
    if not actions:
        return
    st.markdown("🛡️ **Actions requested by the AI agent** (human-in-the-loop)")
    for a in actions:
        with st.container(border=True):
            c1, c2, c3 = st.columns([3, 1, 1])
            args = {k: v for k, v in a["args"].items() if k != "customer_email"}
            manager = " · 👔 needs manager" if a.get("requires_manager") else ""
            c1.markdown(f"`{a['action_id']}` **{a['tool']}** - `{a['status']}`{manager}")
            c1.caption(json.dumps(args, default=str))
            if a["status"] == "PENDING_APPROVAL":
                if c2.button("Approve", key=f"{key_prefix}_ap_{a['action_id']}", type="primary", use_container_width=True):
                    approve_action(a["action_id"])
                    st.rerun()
                if c3.button("Reject", key=f"{key_prefix}_rj_{a['action_id']}", use_container_width=True):
                    reject_action(a["action_id"])
                    st.rerun()
            elif a.get("result"):
                c1.success(f"Result: {a['result']}")


def highlight_evidence(text: str, evidence: list) -> str:
    safe = html.escape(text)
    for quote in evidence:
        pattern = re.compile(re.escape(html.escape(quote.strip())), re.IGNORECASE)
        safe = pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", safe)
    return safe.replace("\n", "<br>")


def shapley_chart(explanation: dict):
    df = pd.DataFrame([
        {"factor": f"{c['label']} = {c['value']}", "impact": c["impact_pct_points"]}
        for c in explanation["contributions"]
    ])
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X("impact:Q", title="Impact on delay risk (percentage points)"),
        y=alt.Y("factor:N", sort=None, title=None),
        color=alt.condition(alt.datum.impact > 0, alt.value("#DC2626"), alt.value("#059669")),
        tooltip=["factor", "impact"],
    ).properties(height=170)
    st.altair_chart(chart, use_container_width=True)


# --- Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2830/2830312.png", width=60)
    st.markdown("### **ShipMate AI Platform**")
    st.caption("Conversational · Agentic · Explainable")
    # Reserved slot: filled at the END of the script run (see render_sidebar_metrics),
    # because Streamlit executes top-to-bottom and tickets are created further down.
    metrics_slot = st.container()


def render_sidebar_metrics():
    with metrics_slot:
        st.divider()
        st.markdown("**System Metrics**")
        st.metric("Total Tickets Processed", len(st.session_state.tickets))
        open_tickets = len([t for t in st.session_state.tickets if t["status"] == "OPEN"])
        st.metric("Pending Open Tickets", open_tickets)
        pending = sum(a["status"] == "PENDING_APPROVAL" for t in st.session_state.tickets for a in t.get("pending_actions", []))
        st.metric("Actions Awaiting Approval", pending)

        st.divider()
        st.caption("Built with Gemini (tool calling) + Scikit-Learn Random Forest + exact Shapley explanations + Streamlit.")

# --- Navigation Tabs ---
tab_customer, tab_chat, tab_agent, tab_copilot = st.tabs(
    ["👤 Customer Support Portal", "💬 AI Chat Assistant", "🎧 Support Agent Workspace", "🧠 Ops Copilot"]
)

# ==============================================================================
# TAB 1: CUSTOMER PORTAL
# ==============================================================================
with tab_customer:
    st.markdown("<div class='main-header'>Submit a Support Ticket</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Our AI engine extracts tracking details, queries carrier hubs, and predicts SLA breach risks.</div>", unsafe_allow_html=True)

    # Preset Loader
    st.markdown("##### ⚡ Quick-Fill Demo Presets")
    preset_cols = st.columns(3)
    selected_preset = None
    if preset_cols[0].button("📦 Delayed Order (SHIP-1001)", use_container_width=True):
        selected_preset = "Delayed Delivery (Potsdam)"
    if preset_cols[1].button("🛃 Customs Hold (SHIP-2002)", use_container_width=True):
        selected_preset = "Customs Exception (Paris CDG)"
    if preset_cols[2].button("💥 Damaged Goods (SHIP-3003)", use_container_width=True):
        selected_preset = "Damaged Goods (Amsterdam)"

    if selected_preset:
        preset_data = DEMO_SCENARIOS[selected_preset]
        st.session_state.form_name = preset_data["name"]
        st.session_state.form_email = preset_data["email"]
        st.session_state.form_subject = preset_data["subject"]
        st.session_state.form_tracking = preset_data["tracking"]
        st.session_state.form_body = preset_data["body"]

    with st.form("customer_ticket_form"):
        col1, col2 = st.columns(2)
        cust_name = col1.text_input("Your Name", value=st.session_state.get("form_name", "Max Mustermann"))
        cust_email = col2.text_input("Your Email", value=st.session_state.get("form_email", "max.mustermann@post.de"))

        col3, col4 = st.columns([2, 1])
        cust_subject = col3.text_input("Subject", value=st.session_state.get("form_subject", "Where is my package? Delivery delayed!"))
        cust_tracking = col4.text_input("Tracking Code (Optional)", value=st.session_state.get("form_tracking", "SHIP-1001"))

        cust_body = st.text_area("Inquiry / Issue Description", value=st.session_state.get("form_body", "Hi, I have been tracking order SHIP-1001 for 3 days and the status has not moved from Potsdam. I need this urgently for a birthday party tomorrow or I will cancel the order."), height=120)

        submitted = st.form_submit_button("🚀 Submit Inquiry", use_container_width=True)

    if submitted:
        with st.spinner("🤖 Triaging ticket through Gemini LLM and scoring ML delay risk..."), llm_errors():
            ticket_record = run_triage(
                sender_name=cust_name,
                sender_email=cust_email,
                subject=cust_subject,
                body=cust_body,
                tracking_number=cust_tracking or None,
            )
            st.session_state.tickets.insert(0, ticket_record)
            st.success(f"✅ Ticket Created Successfully! Assigned ID: **{ticket_record['ticket_id']}**. Go to the **Agent Workspace** tab to review.")

# ==============================================================================
# TAB 2: CONVERSATIONAL AI CHAT ASSISTANT
# ==============================================================================
with tab_chat:
    st.markdown("<div class='main-header'>AI Chat Assistant</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>A multi-turn assistant that remembers the conversation, asks for missing details, "
                "calls tools (carrier lookup, ML risk, policy, CRM) and requests human approval before any real-world action.</div>",
                unsafe_allow_html=True)

    persona_names = {s["name"]: s["email"] for s in DEMO_SCENARIOS.values()}
    pc1, pc2 = st.columns([3, 1])
    persona = pc1.selectbox("Chatting as customer", list(persona_names), key="chat_persona")
    if pc2.button("🔄 New conversation", use_container_width=True) or st.session_state.chat_session is None \
            or st.session_state.chat_session.customer_name != persona:
        st.session_state.chat_session = ChatSession(
            persona, persona_names[persona],
            on_ticket_created=lambda t: st.session_state.tickets.insert(0, t),
        )

    session: ChatSession = st.session_state.chat_session

    if not session.transcript:
        st.caption("Try a starter message:")
        starter_cols = st.columns(len(CHAT_STARTERS))
        for i, starter in enumerate(CHAT_STARTERS):
            if starter_cols[i].button(starter, key=f"starter_{i}", use_container_width=True):
                st.session_state.pending_chat = starter
                st.rerun()

    for msg in session.transcript:
        with st.chat_message("user" if msg["role"] == "customer" else "assistant"):
            st.markdown(msg["text"])
            if msg["role"] == "assistant":
                render_trace(msg.get("trace", []))

    if session.handoff_summary:
        st.warning(f"🙋 Handed off to a human agent. Summary: {session.handoff_summary}")
    if session.ticket:
        st.info(f"📨 Conversation converted into ticket **{session.ticket['ticket_id']}**. See the Agent Workspace.")
    render_pending_actions(session.pending_actions, key_prefix="chat")

    user_text = st.chat_input("Type your message...", key="chat_input") or st.session_state.pop("pending_chat", None)
    if user_text:
        with st.spinner("🤖 Thinking and calling tools..."), llm_errors():
            session.send(user_text)
        st.rerun()

# ==============================================================================
# TAB 3: AGENT WORKSPACE
# ==============================================================================
with tab_agent:
    st.markdown("<div class='main-header'>Support Agent Workspace</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Live queue with real-time carrier telemetry, explainable ML risk, AI investigations and draft generation.</div>", unsafe_allow_html=True)

    if not st.session_state.tickets:
        st.info("No tickets in the queue yet. Submit a ticket from the Customer Support Portal or chat with the AI Chat Assistant.")
    else:
        col_queue, col_dossier = st.columns([1, 1.6])

        # --- LEFT: Live Queue List ---
        with col_queue:
            st.markdown(f"#### 📥 Incoming Queue ({len(st.session_state.tickets)})")

            for t in st.session_state.tickets:
                with st.container(border=True):
                    header_c1, header_c2 = st.columns([2, 1])
                    source_icon = "💬" if t.get("source") == "chat" else "📝"
                    header_c1.markdown(f"{source_icon} **`{t['ticket_id']}`** — {t['created_at']}")
                    urg_color = "red" if t["analysis"]["urgency"] in ["high", "critical"] else "blue"
                    header_c2.markdown(f":{urg_color}[**{t['analysis']['urgency'].upper()}**] | `{t['status']}`")

                    st.markdown(f"**{t['subject']}**")
                    st.caption(f"{t['customer_name']} ({t['customer_email']})")
                    waiting = sum(a["status"] == "PENDING_APPROVAL" for a in t.get("pending_actions", []))
                    if waiting:
                        st.caption(f"⏸️ {waiting} action(s) awaiting approval")
                    if (t.get("carrier_status") or {}).get("status") in ("NOT_FOUND", "NO_TRACKING_PROVIDED"):
                        st.caption("⚠️ No shipment found - ask the customer for a valid tracking number")

                    if st.button("Inspect Dossier & Draft", key=f"btn_{t['ticket_id']}", use_container_width=True):
                        st.session_state.selected_ticket_id = t["ticket_id"]
                        st.rerun()

        # Selected Ticket Reference
        sel_id = st.session_state.get("selected_ticket_id", st.session_state.tickets[0]["ticket_id"])
        selected_ticket = next((t for t in st.session_state.tickets if t["ticket_id"] == sel_id), st.session_state.tickets[0])
        analysis = selected_ticket["analysis"]

        # --- RIGHT: Full AI & ML Dossier ---
        with col_dossier:
            st.markdown(f"#### 🔎 Ticket Dossier: `{selected_ticket['ticket_id']}`")

            with st.container(border=True):
                st.markdown(f"**Subject:** {selected_ticket['subject']}")
                st.caption(f"**From:** {selected_ticket['customer_name']} <{selected_ticket['customer_email']}> · "
                           f"**Category:** `{analysis['category']}` · **Language:** `{analysis.get('language', 'en')}`")
                st.markdown("💬 **Customer Message** (highlighted = evidence the AI used)")
                st.markdown(f"<div class='customer-msg'>{highlight_evidence(selected_ticket['body'], analysis.get('evidence', []))}</div>",
                            unsafe_allow_html=True)
                if selected_ticket.get("transcript"):
                    with st.expander("🗨️ Full chat transcript"):
                        for m in selected_ticket["transcript"]:
                            who = "Customer" if m["role"] == "customer" else "ShipMate AI"
                            st.markdown(f"**{who}:** {m['text']}")

            # 3-Column Telemetry
            m1, m2, m3 = st.columns(3)
            with m1:
                with st.container(border=True):
                    st.markdown("🚚 **Carrier Telemetry**")
                    cs = selected_ticket["carrier_status"] or {}
                    st.caption(f"**Status:** `{cs.get('status')}`")
                    if cs.get("status") in ("NOT_FOUND", "NO_TRACKING_PROVIDED"):
                        st.caption(cs.get("notes", "No carrier record found."))
                    else:
                        st.caption(f"**Carrier:** {cs.get('carrier')}")
                        st.caption(f"**Route:** {cs.get('origin')} → {cs.get('destination')}")
                        st.caption(f"**Last hub:** {cs.get('last_hub')}")
                        if cs.get("dwell_time_hours") is not None:
                            st.caption(f"**Last scan:** {cs['dwell_time_hours']}h ago")
                        if cs.get("delivered_at"):
                            st.caption(f"**Delivered:** {cs['delivered_at']} ({cs.get('signed_by')})")

            with m2:
                with st.container(border=True):
                    st.markdown("📊 **ML Delay Probability**")
                    ml = selected_ticket["ml_prediction"]
                    if ml:
                        prob = ml["delay_probability"]
                        st.metric("SLA Breach Risk", f"{prob}%")
                        st.progress(prob / 100.0)
                        st.caption(f"Model: {ml['model_used']}")
                    else:
                        st.metric("SLA Breach Risk", "N/A")
                        st.caption(selected_ticket.get("ml_not_applicable") or "Not applicable.")

            with m3:
                with st.container(border=True):
                    st.markdown("🤖 **AI Directive**")
                    st.warning(analysis["action_required"])

            if cs.get("scan_events"):
                with st.expander(f"📍 Carrier scan history ({len(cs['scan_events'])} events)"):
                    st.dataframe(pd.DataFrame(cs["scan_events"]), hide_index=True, use_container_width=True)

            # --- Explainable AI ---
            with st.container(border=True):
                st.markdown("🔍 **Why did the AI decide this?**")
                x_ml, x_cf, x_llm, x_dec = st.tabs(["ML risk factors", "What-if", "LLM reasoning", "Escalation trace"])
                explanation = selected_ticket.get("ml_explanation")

                with x_ml:
                    if explanation:
                        st.caption(f"Average risk {explanation['base_value']}% → this shipment {explanation['prediction']}%. "
                                   f"{explanation['method']}.")
                        shapley_chart(explanation)
                        for w in explanation["warnings"]:
                            st.error(f"⚠️ Data quality: {w}")
                        for n in explanation.get("notes", []):
                            st.caption(f"ℹ️ {n}")
                        if selected_ticket.get("ml_narrative"):
                            st.info(selected_ticket["ml_narrative"])
                        elif st.button("🗣️ Explain in plain English", key=f"narrate_{selected_ticket['ticket_id']}"):
                            with st.spinner("Translating the model's reasoning..."), llm_errors():
                                selected_ticket["ml_narrative"] = narrate_risk_explanation(ml, explanation)
                            st.rerun()
                    else:
                        st.info(f"The delay model was not run: {selected_ticket.get('ml_not_applicable')}")

                with x_cf:
                    if explanation:
                        for cf in explanation["counterfactuals"]:
                            st.markdown(f"- {cf}")
                    else:
                        st.caption("No what-if analysis: the delay model was not run for this ticket.")

                with x_llm:
                    st.markdown(f"**Urgency:** `{analysis['urgency'].upper()}`")
                    st.markdown(analysis.get("urgency_reasoning") or "_No reasoning returned._")
                    if analysis.get("evidence"):
                        st.markdown("**Verbatim evidence** (quotes not found in the ticket are discarded automatically):")
                        for q in analysis["evidence"]:
                            st.markdown(f"> {q}")

                with x_dec:
                    reasons = selected_ticket.get("escalation_reasons", [])
                    if reasons:
                        st.markdown(f"**Decision:** `{selected_ticket.get('action_taken')}` because:")
                        for r in reasons:
                            st.markdown(f"- ✅ {r}")
                    else:
                        st.markdown("**Decision:** `AUTO_DRAFT_CREATED` - no escalation rule fired "
                                    "(ML risk below 70% or not applicable, no customs hold, urgency below HIGH).")

            # --- Agentic AI ---
            with st.container(border=True):
                st.markdown("🤖 **AI Investigation Agent**")
                st.caption("The agent decides which tools to call (carrier, ML risk, CRM history, policy) and may request actions for your approval.")
                if st.button("▶️ Run AI investigation", key=f"inv_{selected_ticket['ticket_id']}"):
                    with st.spinner("Agent is investigating..."), llm_errors():
                        result = investigate_ticket(selected_ticket)
                    selected_ticket["agent_summary"] = result["reply"]
                    selected_ticket.setdefault("agent_trace", []).extend(result["trace"])
                    selected_ticket.setdefault("pending_actions", []).extend(result["new_actions"])
                    st.rerun()
                if selected_ticket.get("handoff_summary"):
                    st.warning(f"🙋 Chat handoff summary: {selected_ticket['handoff_summary']}")
                if selected_ticket.get("agent_summary"):
                    st.markdown(selected_ticket["agent_summary"])
                render_trace(selected_ticket.get("agent_trace", []))
                render_pending_actions(selected_ticket.get("pending_actions", []), key_prefix=f"ws_{selected_ticket['ticket_id']}")

            # AI Draft Reply Area
            with st.container(border=True):
                st.markdown("✨ **AI Auto-Drafted Customer Reply**")
                agent_reply_text = st.text_area(
                    "Review or edit before dispatching:",
                    value=selected_ticket["draft_reply"],
                    height=160,
                    key=f"draft_{selected_ticket['ticket_id']}"
                )

                btn_c1, btn_c2 = st.columns(2)
                if btn_c1.button("🚨 Escalate to Lead", key="btn_esc", use_container_width=True):
                    selected_ticket["status"] = "ESCALATED"
                    st.toast(f"Ticket {selected_ticket['ticket_id']} escalated to lead!", icon="🚨")
                    st.rerun()

                if btn_c2.button("✅ Approve & Dispatch", key="btn_app", type="primary", use_container_width=True):
                    selected_ticket["status"] = "RESOLVED"
                    selected_ticket["draft_reply"] = agent_reply_text
                    st.toast(f"Reply sent for {selected_ticket['ticket_id']}!", icon="✅")
                    st.rerun()

# ==============================================================================
# TAB 4: OPS COPILOT
# ==============================================================================
with tab_copilot:
    st.markdown("<div class='main-header'>Ops Copilot</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Ask questions about the support queue in plain language. "
                "The copilot answers only from live ticket data via tools.</div>", unsafe_allow_html=True)

    if st.button("🧹 Clear copilot conversation"):
        st.session_state.copilot_contents = []
        st.session_state.copilot_log = []
        st.rerun()

    if not st.session_state.copilot_log:
        cols = st.columns(len(COPILOT_STARTERS))
        for i, q in enumerate(COPILOT_STARTERS):
            if cols[i].button(q, key=f"cop_starter_{i}", use_container_width=True):
                st.session_state.pending_copilot = q
                st.rerun()

    for entry in st.session_state.copilot_log:
        with st.chat_message("user"):
            st.markdown(entry["question"])
        with st.chat_message("assistant"):
            st.markdown(entry["answer"])
            render_trace(entry["trace"], title="Copilot tool calls")

    question = st.chat_input("Ask the copilot...", key="copilot_input") or st.session_state.pop("pending_copilot", None)
    if question:
        contents = st.session_state.copilot_contents
        checkpoint = len(contents)
        contents.append(types.Content(role="user", parts=[types.Part(text=question)]))
        with st.spinner("Querying the ticket queue..."), llm_errors():
            try:
                result = build_copilot_agent(lambda: st.session_state.tickets).run(contents)
            except LLMUnavailableError:
                del contents[checkpoint:]  # roll back so the history stays valid for a retry
                raise
        st.session_state.copilot_log.append({"question": question, "answer": result["reply"], "trace": result["trace"]})
        st.rerun()

# Rendered last so the metrics reflect tickets created during this run
render_sidebar_metrics()
