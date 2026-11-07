"""Tools for the example support agent.

Each is a plain, typed Python function with a Google-style docstring. The
``@tool`` decorator turns it into a portable ``ToolSpec``; the same function
runs inside LangGraph (local + Agent Engine) and as a Claude Managed Agents
custom tool.
"""

from __future__ import annotations

import hashlib
from datetime import date

from mfa import tool

from agents.support_agent.data import ORDERS, POLICIES, TODAY

REFUND_WINDOW_DAYS = 30


@tool
def get_order_status(order_id: str) -> dict:
    """Look up an order's status, items, carrier and tracking number.

    Args:
        order_id: Order identifier in the form ORD-1234.
    """
    order = ORDERS.get(order_id.strip().upper())
    if order is None:
        return {"found": False, "order_id": order_id, "message": "No order with that ID."}
    return {"found": True, "order_id": order_id.strip().upper(), **order}


@tool
def check_refund_eligibility(order_id: str) -> dict:
    """Check whether an order can be refunded under the 30-day returns policy, and for how much.

    Args:
        order_id: Order identifier in the form ORD-1234.
    """
    oid = order_id.strip().upper()
    order = ORDERS.get(oid)
    if order is None:
        return {"order_id": oid, "eligible": False, "reason": "Order not found."}
    total = round(sum(i["price"] for i in order["items"]), 2)
    if order["status"] == "processing":
        return {
            "order_id": oid,
            "eligible": True,
            "amount": total,
            "reason": "Order has not shipped; it can be cancelled for a full refund.",
        }
    if order["delivered"] is None:
        return {
            "order_id": oid,
            "eligible": False,
            "reason": "Order is in transit; a return can be started once it is delivered.",
        }
    days = (TODAY - date.fromisoformat(order["delivered"])).days
    if days <= REFUND_WINDOW_DAYS:
        return {
            "order_id": oid,
            "eligible": True,
            "amount": total,
            "days_since_delivery": days,
            "reason": f"Delivered {days} days ago, within the {REFUND_WINDOW_DAYS}-day window.",
        }
    return {
        "order_id": oid,
        "eligible": False,
        "days_since_delivery": days,
        "reason": f"Delivered {days} days ago, outside the {REFUND_WINDOW_DAYS}-day window.",
    }


@tool
def search_policies(query: str) -> list[dict]:
    """Search the store's policy knowledge base (returns, shipping, warranty, cancellations, price match).

    Args:
        query: Keywords describing the policy question.
    """
    words = {w.strip("?.,!").lower() for w in query.split() if len(w) > 2}
    scored = []
    for topic, text in POLICIES.items():
        hay = f"{topic.replace('_', ' ')} {text}".lower()
        score = sum(1 for w in words if w in hay)
        if score:
            scored.append((score, topic, text))
    scored.sort(reverse=True)
    return [{"topic": t, "text": txt} for _, t, txt in scored[:2]] or [
        {"topic": None, "text": "No matching policy found."}
    ]


@tool
def create_support_ticket(summary: str, priority: str, order_id: str = "") -> dict:
    """Escalate an issue to a human support specialist by opening a ticket.

    Args:
        summary: One or two sentences describing the customer's problem.
        priority: One of "low", "normal" or "urgent".
        order_id: Related order identifier, if any.
    """
    priority = priority.lower()
    if priority not in {"low", "normal", "urgent"}:
        return {"created": False, "error": "priority must be low, normal or urgent"}
    digest = hashlib.sha1(f"{order_id}|{summary}".encode()).hexdigest()[:6].upper()
    return {
        "created": True,
        "ticket_id": f"TCK-{digest}",
        "priority": priority,
        "order_id": order_id or None,
        "eta": "within 4 hours" if priority == "urgent" else "within 1 business day",
    }


TOOLS = [get_order_status, check_refund_eligibility, search_policies, create_support_ticket]
