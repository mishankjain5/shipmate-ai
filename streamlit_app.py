import streamlit as st
import pandas as pd
import uuid
from datetime import datetime
import json
import os

# Internal project modules
from app.schemas import TicketPayload, UrgencyLevel
from app.llm_agent import analyze_ticket, generate_draft_response
from app.mock_carrier_api import lookup_shipment
from app.ml_delay_model import predict_delay_risk

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
</style>
""", unsafe_allow_html=True)

# --- State Management ---
if "tickets" not in st.session_state:
    st.session_state.tickets = []

DEMO_SCENARIOS = {
    "Delayed Delivery (Potsdam)": {
        "name": "Max Mustermann",
        "email": "max.mustermann@post.de",
        "subject": "Where is my package? Delivery delayed!",
        "tracking": "SEVEN-1001",
        "body": "Hi, I have been tracking order SEVEN-1001 for 3 days and the status has not moved from Potsdam. I need this urgently for a birthday party tomorrow or I will cancel the order."
    },
    "Customs Exception (Paris CDG)": {
        "name": "Claire Dupont",
        "email": "c.dupont@paris-store.fr",
        "subject": "Colis bloqué à la douane / Customs Hold",
        "tracking": "SEVEN-2002",
        "body": "Hello, my shipment SEVEN-2002 to Paris is marked with a customs hold exception at CDG airport. Can you check what documents are missing to clear this?"
    },
    "Damaged Goods (Amsterdam)": {
        "name": "Jan de Vries",
        "email": "jan.vries@enterprise.nl",
        "subject": "DAMAGED GOODS ON ARRIVAL - REFUND NEEDED",
        "tracking": "SEVEN-3003",
        "body": "Our package SEVEN-3003 arrived completely crushed, torn open, and unusable. Please process an immediate replacement or full refund right away."
    }
}

# --- Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2830/2830312.png", width=60)
    st.markdown("### **ShipMate AI Platform**")
    st.caption("Hybrid LLM & Scikit-Learn Triage Engine")
    
    st.divider()
    st.markdown("**System Metrics**")
    st.metric("Total Tickets Processed", len(st.session_state.tickets))
    open_tickets = len([t for t in st.session_state.tickets if t["status"] == "OPEN"])
    st.metric("Pending Open Tickets", open_tickets)
    
    st.divider()
    st.caption("Built with Gemini Flash + Scikit-Learn Random Forest + Streamlit.")

# --- Navigation Tabs ---
tab_customer, tab_agent = st.tabs(["👤 Customer Support Portal", "🎧 Support Agent Workspace"])

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
    if preset_cols[0].button("📦 Delayed Order (SEVEN-1001)", use_container_width=True):
        selected_preset = "Delayed Delivery (Potsdam)"
    if preset_cols[1].button("🛃 Customs Hold (SEVEN-2002)", use_container_width=True):
        selected_preset = "Customs Exception (Paris CDG)"
    if preset_cols[2].button("💥 Damaged Goods (SEVEN-3003)", use_container_width=True):
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
        cust_tracking = col4.text_input("Tracking Code (Optional)", value=st.session_state.get("form_tracking", "SEVEN-1001"))

        cust_body = st.text_area("Inquiry / Issue Description", value=st.session_state.get("form_body", "Hi, I have been tracking order SEVEN-1001 for 3 days and the status has not moved from Potsdam. I need this urgently for a birthday party tomorrow or I will cancel the order."), height=120)

        submitted = st.form_submit_button("🚀 Submit Inquiry", use_container_width=True)

    if submitted:
        with st.spinner("🤖 Triaging ticket through Gemini LLM and scoring ML delay risk..."):
            ticket_id = f"TICK-{uuid.uuid4().hex[:6].upper()}"
            payload = TicketPayload(
                ticket_id=ticket_id,
                sender_email=cust_email,
                subject=cust_subject,
                body=cust_body
            )

            # 1. LLM Extraction
            analysis = analyze_ticket(payload)
            tracking_id = cust_tracking or analysis.tracking_number
            if tracking_id:
                analysis.tracking_number = tracking_id

            # 2. Carrier Telemetry
            carrier_info = lookup_shipment(tracking_id)

            # 3. Tabular ML Risk Prediction
            carrier_name = carrier_info.get("carrier", "DHL Express") if carrier_info else "DHL Express"
            hub_name = carrier_info.get("last_hub", "Potsdam Sorting Facility") if carrier_info else "Potsdam Sorting Facility"
            simulated_dwell = 44.0 if (carrier_info and carrier_info.get("status") in ["IN_TRANSIT", "CUSTOMS_HOLD"]) else 12.0

            ml_res = predict_delay_risk(
                carrier=carrier_name,
                last_hub=hub_name,
                dwell_time_hours=simulated_dwell,
                is_cross_border=1
            )

            # 4. Draft Reply
            draft = generate_draft_response(payload, analysis, carrier_info or {"note": "No carrier records"})

            ticket_record = {
                "ticket_id": ticket_id,
                "created_at": datetime.now().strftime("%H:%M:%S"),
                "customer_name": cust_name,
                "customer_email": cust_email,
                "subject": cust_subject,
                "body": cust_body,
                "analysis": analysis.model_dump(),
                "carrier_status": carrier_info,
                "ml_prediction": ml_res,
                "draft_reply": draft,
                "status": "OPEN"
            }

            st.session_state.tickets.insert(0, ticket_record)
            st.success(f"✅ Ticket Created Successfully! Assigned ID: **{ticket_id}**. Go to the **Agent Workspace** tab to review.")

# ==============================================================================
# TAB 2: AGENT WORKSPACE
# ==============================================================================
with tab_agent:
    st.markdown("<div class='main-header'>Support Agent Workspace</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Live queue with real-time carrier telemetry, ML SLA breach risk, and AI draft generation.</div>", unsafe_allow_html=True)

    if not st.session_state.tickets:
        st.info("No tickets in the queue yet. Submit a test ticket from the Customer Support Portal tab.")
    else:
        col_queue, col_dossier = st.columns([1, 1.6])

        # --- LEFT: Live Queue List ---
        with col_queue:
            st.markdown(f"#### 📥 Incoming Queue ({len(st.session_state.tickets)})")
            
            selected_ticket = None
            for idx, t in enumerate(st.session_state.tickets):
                is_selected = st.session_state.get("selected_ticket_id") == t["ticket_id"] or idx == 0
                
                with st.container(border=True):
                    header_c1, header_c2 = st.columns([2, 1])
                    header_c1.markdown(f"**`{t['ticket_id']}`** — {t['created_at']}")
                    urg_color = "red" if t["analysis"]["urgency"] in ["high", "critical"] else "blue"
                    header_c2.markdown(f":{urg_color}[**{t['analysis']['urgency'].upper()}**] | `{t['status']}`")

                    st.markdown(f"**{t['subject']}**")
                    st.caption(f"{t['customer_name']} ({t['customer_email']})")

                    if st.button("Inspect Dossier & Draft", key=f"btn_{t['ticket_id']}", use_container_width=True):
                        st.session_state.selected_ticket_id = t["ticket_id"]
                        st.rerun()

        # Selected Ticket Reference
        sel_id = st.session_state.get("selected_ticket_id", st.session_state.tickets[0]["ticket_id"])
        selected_ticket = next((t for t in st.session_state.tickets if t["ticket_id"] == sel_id), st.session_state.tickets[0])

        # --- RIGHT: Full AI & ML Dossier ---
        with col_dossier:
            st.markdown(f"#### 🔎 Ticket Dossier: `{selected_ticket['ticket_id']}`")
            
            with st.container(border=True):
                st.markdown(f"**Subject:** {selected_ticket['subject']}")
                st.caption(f"**From:** {selected_ticket['customer_name']} <{selected_ticket['customer_email']}>")
                st.info(f"💬 **Customer Message:**\n\n{selected_ticket['body']}")

            # 3-Column Telemetry
            m1, m2, m3 = st.columns(3)
            with m1:
                with st.container(border=True):
                    st.markdown("🚚 **Carrier Telemetry**")
                    if selected_ticket["carrier_status"]:
                        st.caption(f"**Status:** `{selected_ticket['carrier_status'].get('status')}`")
                        st.caption(f"**Carrier:** {selected_ticket['carrier_status'].get('carrier')}")
                        st.caption(f"**Hub:** {selected_ticket['carrier_status'].get('last_hub')}")
                    else:
                        st.caption("No carrier record found.")

            with m2:
                with st.container(border=True):
                    st.markdown("📊 **ML Delay Probability**")
                    ml = selected_ticket["ml_prediction"]
                    prob = ml["delay_probability"]
                    st.metric("SLA Breach Risk", f"{prob}%")
                    st.progress(prob / 100.0)
                    st.caption(f"Model: {ml['model_used']}")

            with m3:
                with st.container(border=True):
                    st.markdown("🤖 **AI Directive**")
                    st.warning(selected_ticket["analysis"]["action_required"])

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