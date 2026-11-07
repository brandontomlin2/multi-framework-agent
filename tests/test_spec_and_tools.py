import json

from mfa.registry import load_agent
from mfa.spec import AGENT_ENGINE, CLAUDE_MANAGED, LOCAL


def test_agent_loads_by_short_name_and_module_ref():
    a = load_agent("support_agent")
    b = load_agent("agents.support_agent.agent:AGENT")
    assert a is b
    assert {t.name for t in a.tools} == {
        "get_order_status",
        "check_refund_eligibility",
        "search_policies",
        "create_support_ticket",
    }
    for target in (LOCAL, AGENT_ENGINE, CLAUDE_MANAGED):
        assert a.model_for(target)


def test_tool_schema_is_derived_from_signature_and_docstring():
    spec = load_agent("support_agent")
    t = spec.tool_by_name("create_support_ticket")
    assert t.input_schema["type"] == "object"
    assert set(t.input_schema["required"]) == {"summary", "priority"}
    assert "order identifier" in t.input_schema["properties"]["order_id"]["description"]
    assert "title" not in t.input_schema


def test_tools_are_deterministic():
    spec = load_agent("support_agent")
    out = json.loads(spec.tool_by_name("check_refund_eligibility").invoke({"order_id": "ORD-1002"}))
    assert out["eligible"] is True and out["amount"] == 215.0
    out = json.loads(spec.tool_by_name("check_refund_eligibility").invoke({"order_id": "ord-1003"}))
    assert out["eligible"] is False
    hits = json.loads(spec.tool_by_name("search_policies").invoke({"query": "standard shipping free"}))
    assert hits[0]["topic"] == "shipping"
