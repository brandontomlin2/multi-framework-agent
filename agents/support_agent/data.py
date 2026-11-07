"""Fixture data for the example agent. Swap for real systems (DB, API) in your own agent."""

from datetime import date

TODAY = date(2026, 11, 7)  # fixed so refund-window evals are deterministic

ORDERS = {
    "ORD-1001": {
        "customer": "Dana Lee",
        "items": [{"sku": "TENT-2P", "name": "Ridgeline 2-Person Tent", "price": 249.00}],
        "status": "shipped",
        "carrier": "UPS",
        "tracking": "1Z999AA10123456784",
        "ordered": "2026-10-30",
        "delivered": None,
    },
    "ORD-1002": {
        "customer": "Sam Ortiz",
        "items": [
            {"sku": "BOOT-M10", "name": "Summit Hiking Boots (M10)", "price": 179.00},
            {"sku": "SOCK-3PK", "name": "Merino Socks 3-pack", "price": 36.00},
        ],
        "status": "delivered",
        "carrier": "FedEx",
        "tracking": "612999AA1012",
        "ordered": "2026-10-20",
        "delivered": "2026-10-25",
    },
    "ORD-1003": {
        "customer": "Priya Shah",
        "items": [{"sku": "JKT-RAIN-L", "name": "Stormshell Rain Jacket (L)", "price": 199.00}],
        "status": "delivered",
        "carrier": "USPS",
        "tracking": "9400111899223",
        "ordered": "2026-08-15",
        "delivered": "2026-08-20",
    },
    "ORD-1004": {
        "customer": "Alex Kim",
        "items": [{"sku": "STOVE-1", "name": "Pocket Camp Stove", "price": 59.00}],
        "status": "processing",
        "carrier": None,
        "tracking": None,
        "ordered": "2026-11-06",
        "delivered": None,
    },
}

POLICIES = {
    "returns": (
        "Returns: unused items may be returned within 30 days of delivery for a full refund to "
        "the original payment method. Items returned after 30 days are not eligible for a refund. "
        "Return shipping is free for exchanges and $7.95 for refunds."
    ),
    "shipping": (
        "Shipping: standard shipping takes 3-7 business days and is free on orders over $75. "
        "Expedited shipping (1-2 business days) costs $19.95. We ship to the US and Canada only."
    ),
    "warranty": (
        "Warranty: tents, jackets and boots carry a 2-year warranty against manufacturing defects. "
        "Normal wear and tear and accidental damage are not covered."
    ),
    "cancellations": (
        "Cancellations: orders can be cancelled free of charge while they are still processing. "
        "Once shipped, an order must be returned instead."
    ),
    "price_match": (
        "Price match: we match the price of identical items from authorized US retailers within "
        "14 days of purchase. Marketplace sellers and clearance sales are excluded."
    ),
}
