"""Agent loop tests with a scripted fake Gemini client (no network, no API key)."""
from types import SimpleNamespace

from google.genai import types

from app.agent import Agent, PENDING_ACTIONS, approve_action, reject_action, build_support_tools, build_copilot_tools


def _call(name, **args):
    return SimpleNamespace(name=name, args=args, id=None)


def _response(calls=None, text=None):
    parts = [types.Part(text=text)] if text else [types.Part(text="(tool call)")]
    return SimpleNamespace(
        candidates=[SimpleNamespace(content=types.Content(role="model", parts=parts))],
        function_calls=calls,
        text=text,
    )


class FakeClient:
    """Returns pre-scripted responses in order and records every request."""
    def __init__(self, script):
        self.script = list(script)
        self.requests = []
        self.models = self

    def generate_content(self, model, contents, config):
        self.requests.append(list(contents))
        return self.script.pop(0)


def _user(text):
    return [types.Content(role="user", parts=[types.Part(text=text)])]


def test_read_only_tool_runs_then_agent_answers():
    client = FakeClient([
        _response(calls=[_call("lookup_shipment", tracking_number="SHIP-2002")]),
        _response(text="Your parcel is held at customs."),
    ])
    agent = Agent("sys", build_support_tools("c.dupont@paris-store.fr"), client=client)
    result = agent.run(_user("Where is SHIP-2002?"))

    assert result["reply"] == "Your parcel is held at customs."
    tool_step = result["trace"][0]
    assert tool_step["type"] == "tool" and tool_step["result"]["status"] == "CUSTOMS_HOLD"
    # The tool result was fed back to the model on the second request
    fn_response = client.requests[1][-1].parts[0].function_response
    assert fn_response.name == "lookup_shipment"


def test_side_effecting_tool_is_queued_not_executed():
    client = FakeClient([
        _response(calls=[_call("issue_voucher", amount_eur=80, reason="late parcel")]),
        _response(text="I've requested a voucher for you."),
    ])
    agent = Agent("sys", build_support_tools("max.mustermann@post.de"), client=client)
    result = agent.run(_user("I want compensation"))

    action = result["new_actions"][0]
    assert action["status"] == "PENDING_APPROVAL"
    assert action["requires_manager"] is True                     # > EUR 50
    assert action["args"]["customer_email"] == "max.mustermann@post.de"  # bound server-side
    assert action["result"] is None

    approved = approve_action(action["action_id"])
    assert approved["status"] == "EXECUTED"
    assert approved["result"]["voucher_code"].startswith("SM-")


def test_rejected_action_never_executes():
    client = FakeClient([
        _response(calls=[_call("request_customs_documents", tracking_number="SHIP-2002", documents=["commercial invoice"])]),
        _response(text="Requested."),
    ])
    result = Agent("sys", build_support_tools("c.dupont@paris-store.fr"), client=client).run(_user("help"))
    action_id = result["new_actions"][0]["action_id"]
    assert reject_action(action_id)["status"] == "REJECTED"
    assert approve_action(action_id)["status"] == "REJECTED"   # cannot approve after rejection
    assert PENDING_ACTIONS[action_id]["result"] is None


def test_voucher_guardrail_blocks_excessive_amount():
    client = FakeClient([
        _response(calls=[_call("issue_voucher", amount_eur=5000, reason="angry")]),
        _response(text="I can't do that, let me get a colleague."),
    ])
    result = Agent("sys", build_support_tools("x@y.z"), client=client).run(_user("give me 5000"))
    assert result["new_actions"] == []
    assert result["trace"][0]["type"] == "blocked"


def test_customer_history_is_bound_to_session_customer():
    """The model cannot pass another customer's email: the tool has no email parameter."""
    client = FakeClient([
        _response(calls=[_call("get_customer_history")]),
        _response(text="ok"),
    ])
    result = Agent("sys", build_support_tools("jan.vries@enterprise.nl"), client=client).run(_user("hi"))
    assert result["trace"][0]["result"]["segment"] == "B2B enterprise"


def test_step_limit_stops_runaway_loop():
    looping = [_response(calls=[_call("check_policy", topic="customs")]) for _ in range(3)]
    result = Agent("sys", build_support_tools("x@y.z"), client=FakeClient(looping), max_steps=3).run(_user("loop"))
    assert result["trace"][-1]["type"] == "limit"


def test_copilot_search_filters_tickets():
    tickets = [
        {"ticket_id": "T1", "created_at": "", "customer_name": "A", "subject": "s", "status": "OPEN", "body": "",
         "analysis": {"category": "delayed_delivery", "urgency": "high", "summary": ""},
         "carrier_status": {"status": "IN_TRANSIT"}, "ml_prediction": {"delay_probability": 86.0},
         "escalation_reasons": ["x"], "pending_actions": []},
        {"ticket_id": "T2", "created_at": "", "customer_name": "B", "subject": "s", "status": "RESOLVED", "body": "",
         "analysis": {"category": "damaged_item", "urgency": "low", "summary": ""},
         "carrier_status": {"status": "DELIVERED"}, "ml_prediction": {"delay_probability": 3.9},
         "escalation_reasons": [], "pending_actions": []},
    ]
    tools = {t.name: t for t in build_copilot_tools(lambda: tickets)}
    assert tools["search_tickets"].func(min_delay_risk=70)["count"] == 1
    assert tools["queue_statistics"].func()["escalated"] == 1
