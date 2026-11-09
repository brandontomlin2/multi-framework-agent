"""The second example agent: richer argument shapes, a two-step mutation and policy decisions, on every target."""

from __future__ import annotations

import json

import pytest

from agents.order_support_agent import orders
from agents.order_support_agent.agent import AGENT as ORDER_SPEC
from mfa.evals.dataset import load_cases
from mfa.evals.harness import run_eval
from mfa.spec import AGENT_ENGINE, CLAUDE_MANAGED, CLOUD_RUN, LOCAL

ALL_TARGETS = (LOCAL, AGENT_ENGINE, CLOUD_RUN, CLAUDE_MANAGED)
TOOL = {t.name: t for t in ORDER_SPEC.tools}


@pytest.fixture(autouse=True)
def _fresh_store(monkeypatch):
    orders.reset_state()
    monkeypatch.delenv("MFA_ORDERS_FAULTS", raising=False)


def test_lookup_sentinel_and_not_found():
    latest = orders.lookup_order("")
    assert latest["order_id"] == "ORD-1001" and latest["selected_as"] == "latest order" and latest["days_since_delivery"] == 2
    assert latest["return_window_closes"] == "2026-12-07"
    assert orders.lookup_order("latest")["order_id"] == "ORD-1001" and orders.lookup_order("ord-0950")["status"] == "in_transit"
    missing = orders.lookup_order("ORD-7777")
    assert missing["found"] is False and missing["recent_orders"][0] == "ORD-1001" and "ORD-1234" in missing["hint"]


def test_search_filters_and_pagination():
    r = orders.search_products("running shoes", category="shoes", sizes=["9", "10"], max_price=100)
    assert [p["sku"] for p in r["results"]] == ["SH-RUN-9", "SH-SAN-1"] and r["has_more"] is False
    page1 = orders.search_products("jacket")
    page2 = orders.search_products("jacket", page=2)
    assert page1["total"] == 4 and len(page1["results"]) == 3 and page1["has_more"] and page1["next_page"] == 2
    assert [p["sku"] for p in page2["results"]] == ["AP-JKT-2"] or page2["total"] == 4 and len(page2["results"]) == 1
    assert page2["has_more"] is False and page2["next_page"] is None
    nothing = orders.search_products("unicycle")
    assert nothing["results"] == [] and "shoes" in nothing["suggestions"]


def test_return_policy_outcomes():
    ok = orders.start_return("ORD-1001", "SH-RUN-9", "too_small")
    assert ok["status"] == "needs_confirmation" and ok["confirmation_token"].startswith("rtok-")
    assert ok["refund_estimate"] == {"item_total": 89.0, "restocking_fee": 0.0, "return_label_fee": 6.95, "refund": 82.05,
                                     "currency": "USD", "method": "original payment method",
                                     "timing": "5-7 business days after the item reaches the warehouse"}
    free = orders.start_return("ORD-1001", "AP-SOCK-1", "damaged")
    assert free["refund_estimate"]["return_label_fee"] == 0.0 and free["refund_estimate"]["refund"] == 48.0
    window = orders.start_return("ORD-0987", "EL-HP-1", "no_longer_needed")
    assert window["status"] == "rejected" and window["reason"] == "return_window_closed" and window["days_since_delivery"] == 45
    assert any("warranty" in a for a in window["alternatives"])
    final = orders.start_return("ORD-0901", "SH-SAN-1", "other")
    assert final["status"] == "rejected" and final["reason"] == "final_sale"
    assert orders.start_return("ORD-0901", "SH-SAN-1", "damaged")["status"] == "needs_confirmation"  # our fault: allowed
    transit = orders.start_return("ORD-0950", "AP-JKT-1", "no_longer_needed")
    assert transit["status"] == "rejected" and transit["reason"] == "not_delivered" and "2026-11-10" in transit["message"]
    wrong_sku = orders.start_return("ORD-1001", "AP-JKT-1", "other")
    assert wrong_sku["status"] == "not_found" and [i["sku"] for i in wrong_sku["items_in_order"]] == ["SH-RUN-9", "AP-SOCK-1"]
    assert orders.start_return("ORD-9999", "X", "other")["status"] == "not_found"
    with pytest.raises(ValueError):
        orders.start_return("ORD-1001", "SH-RUN-9", "whatever")
    # restocking fee: 10% on "no longer needed" items from $150 (none of the fixture's delivered orders has one, by design)
    assert orders.RESTOCKING_FEE_THRESHOLD == 150.0


def test_confirm_and_check_return_lifecycle():
    token = orders.start_return("ORD-1001", "SH-RUN-9", "too_small")["confirmation_token"]
    assert orders.confirm_return("rtok-nope")["reason"] == "invalid_token"
    pending = orders.confirm_return(token)
    assert pending["status"] == "pending" and pending["ticket_id"] == "RET-5001" and pending["refund"] == 82.05
    again = orders.confirm_return(token)
    assert again["status"] == "already_confirmed" and again["ticket_id"] == "RET-5001"
    assert orders.check_return("RET-5001")["status"] == "pending"
    approved = orders.check_return("ret-5001")
    assert approved["status"] == "approved" and approved["label_url"] == "https://returns.northwind.example/labels/RET-5001.pdf"
    assert orders.check_return("RET-9")["status"] == "not_found"


def test_shipping_estimates():
    fr = orders.shipping_estimate("France", "75001", ["SH-RUN-X"])
    assert (fr["destination"]["country_code"], fr["business_days"], fr["shipping_cost"]) == ("FR", "6-10", 18.0)
    us_free = orders.shipping_estimate("US", "97205", ["SH-RUN-9"])
    assert us_free["shipping_cost"] == 0.0 and "free" in us_free["note"]
    us_small = orders.shipping_estimate("usa", "97205", ["HM-BTL-1"])
    assert us_small["shipping_cost"] == 6.95
    tent = orders.shipping_estimate("Canada", "M5V 2T6", ["HM-TNT-2"], city="Toronto")
    assert tent["shipping_cost"] == 22.0 and tent["oversized_surcharge_applied"] is True
    aq = orders.shipping_estimate("Antarctica", "96599", ["EL-WT-1"])
    assert aq["status"] == "rejected" and aq["reason"] == "country_not_served" and aq["country_code"] == "AQ"
    assert orders.shipping_estimate("FR", "75001", ["NOPE-1"])["status"] == "not_found"


def test_tool_schemas_are_self_contained_and_coerce(monkeypatch):
    schema = TOOL["search_products"].input_schema
    assert "$ref" not in json.dumps(schema) and "$defs" not in schema
    filters = next(s for s in schema["properties"]["filters"]["anyOf"] if s.get("type") == "object")
    assert filters["properties"]["category"]["anyOf"][0]["enum"] == ["shoes", "apparel", "electronics", "home"]
    assert "title" not in filters["properties"]["category"]
    assert TOOL["start_return"].input_schema["properties"]["reason"]["enum"][0] == "damaged"
    # the model may send numbers for sizes and strings for numbers: both are coerced, like `book(when: date)`
    r = json.loads(TOOL["search_products"].run({"query": "shoes", "filters": {"category": "shoes", "sizes": [9, 10], "max_price": "100"}}).text)
    assert r["filters"] == {"category": "shoes", "sizes": ["10", "9"], "max_price": 100.0}
    bad = TOOL["search_products"].run({"query": "x", "filters": {"category": "toys"}})
    assert bad.is_error and bad.text.startswith("Error: invalid arguments for search_products: filters.category")
    nested = json.loads(TOOL["shipping_estimate"].run({"destination": {"country": "FR", "postal_code": "75001"}, "skus": ["SH-RUN-X"]}).text)
    assert nested["shipping_cost"] == 18.0
    # fault injection reaches the model as the shared error text
    monkeypatch.setenv("MFA_ORDERS_FAULTS", "lookup_order")
    fault = TOOL["lookup_order"].run({"order_id": ""})
    assert fault.is_error and fault.text == "Error: tool lookup_order failed: RuntimeError: orders service unavailable (injected fault)"
    assert TOOL["search_products"].run({"query": "jacket"}).is_error is False


RETURN_FLOW = [
    {"tool": "lookup_order", "args": {"order_id": ""}},
    {"tool": "read", "args": {"file_path": "/workspace/skills/returns-policy/SKILL.md"}},
    {"tool": "read", "args": {"file_path": "/workspace/skills/returns-policy/references/policy.md"}},
    {"tool": "start_return", "args": {"order_id": "ORD-1001", "sku": "SH-RUN-9", "reason": "too_small"}},
    {"text": "You'd be returning the Trailblazer running shoes (size 9) because they're too small. After the $6.95 "
             "return-label fee the refund estimate is $82.05 to your original payment method, 5-7 business days after "
             "the shoes reach the warehouse. Shall I submit this return?"},
]


@pytest.mark.parametrize("target", ALL_TARGETS)
def test_return_case_passes_on_every_target(target, make_runner):
    """The suite's hardest single-turn case, driven by a scripted model, passes identically everywhere."""
    runner = make_runner(target, RETURN_FLOW, ORDER_SPEC, "order_support_agent")
    report = run_eval(agent=ORDER_SPEC.name, runner=runner,
                      cases=load_cases(ORDER_SPEC.eval_dataset, ids=["return_needs_confirmation"]), judge=None)
    case = report.cases[0]
    assert case.passed, [(c.name, c.detail) for c in case.checks if c.passed is False]
    assert [c.name for c in case.result.tool_calls] == ["lookup_order", "read", "read", "start_return"]
    assert json.loads(case.result.tool_calls[3].output)["status"] == "needs_confirmation"


@pytest.mark.parametrize("target", ALL_TARGETS)
def test_nested_arguments_and_errors_are_identical_on_every_target(target, make_runner):
    script = [
        {"calls": [
            {"tool": "search_products", "args": {"query": "running shoes", "filters": {"category": "shoes", "sizes": [9, "10"], "max_price": "100"}}},
            {"tool": "shipping_estimate", "args": {"destination": {"country": "France", "postal_code": "75001"}, "skus": ["SH-RUN-X"]}},
            {"tool": "start_return", "args": {"order_id": "ORD-1001", "sku": "SH-RUN-9", "reason": "shrunk"}},
        ]},
        {"text": "done"},
    ]
    r = make_runner(target, script, ORDER_SPEC, "order_support_agent").run("go")
    assert r.output == "done" and r.error is None
    search, ship, bad = r.tool_calls
    assert json.loads(search.output)["results"][0]["sku"] == "SH-RUN-9" and json.loads(search.output)["filters"]["sizes"] == ["10", "9"]
    assert json.loads(ship.output)["shipping_cost"] == 18.0
    assert bad.is_error and bad.output.startswith("Error: invalid arguments for start_return: reason: Input should be")
