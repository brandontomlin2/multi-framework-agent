"""Tools for the example order-support agent.

Compared with the weather agent these exercise harder argument shapes: a sentinel
value (an empty ``order_id`` means the latest order), a nested filter model with an
enum and a list, pagination, an enum reason, a nested address, and a two-step
mutation (``start_return`` then ``confirm_return``) with a pending ticket to poll.
The same functions run inside LangGraph and as Claude Managed Agents custom tools.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from mfa import tool

from agents.order_support_agent import orders

Category = Literal["shoes", "apparel", "electronics", "home"]
ReturnReason = Literal["damaged", "wrong_item", "no_longer_needed", "too_small", "too_large", "other"]


class ProductFilters(BaseModel):
    """Optional filters for a product search."""

    category: Category | None = Field(None, description="Only this product category.")
    sizes: list[str] = Field(default_factory=list, description='Only products available in any of these sizes, e.g. ["9", "10"] or ["M"].')
    max_price: float | None = Field(None, description="Only products at or below this price in USD.")

    @field_validator("sizes", mode="before")
    @classmethod
    def _stringify(cls, v):
        if v is None:
            return []
        if isinstance(v, (str, int, float)):
            v = [v]
        return [str(x).strip() for x in v]


class Address(BaseModel):
    """A shipping destination."""

    country: str = Field(..., description='Country name or ISO code, e.g. "France" or "FR".')
    postal_code: str = Field(..., description="Postal or ZIP code.")
    city: str | None = Field(None, description="City, if known.")


@tool
def lookup_order(order_id: str = "") -> dict:
    """Look up one of the signed-in customer's orders: status, items, dates, tracking and delivery address.

    Args:
        order_id: The order id, e.g. "ORD-1001". Pass an empty string for the customer's latest order.
    """
    return orders.lookup_order(order_id)


@tool
def search_products(query: str, filters: ProductFilters | None = None, page: int = 1) -> dict:
    """Search the product catalogue; results come in pages of three.

    Args:
        query: What the customer is looking for, e.g. "running shoes" or "rain jacket".
        filters: Optional category, sizes and maximum price.
        page: Page number starting at 1; use the result's next_page to continue a search.
    """
    f = filters or ProductFilters()
    return orders.search_products(query, category=f.category, sizes=f.sizes, max_price=f.max_price, page=page)


@tool
def start_return(order_id: str, sku: str, reason: ReturnReason) -> dict:
    """Start a return for one item of a delivered order. Checks the return policy and gives a refund estimate.

    Nothing is submitted yet: the result is either a rejection with the reason and alternatives, or a
    confirmation token to use with confirm_return once the customer has agreed to the summary.

    Args:
        order_id: The order the item belongs to.
        sku: The SKU of the item to return (from the order's items).
        reason: Why the customer is returning it.
    """
    return orders.start_return(order_id, sku, reason)


@tool
def confirm_return(confirmation_token: str) -> dict:
    """Submit a return the customer has explicitly agreed to. Creates a return ticket that is approved asynchronously.

    Args:
        confirmation_token: The token start_return gave for this return.
    """
    return orders.confirm_return(confirmation_token)


@tool
def check_return(ticket_id: str) -> dict:
    """Check a submitted return: pending, or approved with the link to the return label.

    Args:
        ticket_id: The return ticket id, e.g. "RET-5001".
    """
    return orders.check_return(ticket_id)


@tool
def shipping_estimate(destination: Address, skus: list[str]) -> dict:
    """Estimate delivery time and shipping cost for some products to a destination.

    Args:
        destination: Where the parcel would go.
        skus: The SKUs that would be shipped together.
    """
    return orders.shipping_estimate(destination.country, destination.postal_code, skus, destination.city)


TOOLS = [lookup_order, search_products, start_return, confirm_return, check_return, shipping_estimate]
