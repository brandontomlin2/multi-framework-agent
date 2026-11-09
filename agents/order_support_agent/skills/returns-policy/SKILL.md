---
name: returns-policy
description: Northwind's return and refund policy and the step-by-step return procedure. Use whenever a customer wants to return, exchange or get a refund for an item, asks whether something can be returned, or asks about refund timing, restocking fees or return shipping costs.
license: Apache-2.0
metadata:
  author: multi-framework-agent
  version: "1.0"
---

# Returns and refunds

The tools decide; this skill tells you how to run the conversation and how to explain
the decision. `references/policy.md` (in this skill's folder) has the exact rules: read it
before explaining a rejection or a fee.

## Procedure

1. Identify the order and the item. `lookup_order` with an empty `order_id` gives the
   latest order. If the order has several items and the customer has not said which, ask.
2. Call `start_return(order_id, sku, reason)` with the reason that best matches what the
   customer said (`too_small`, `too_large`, `damaged`, `wrong_item`, `no_longer_needed`, `other`).
3. If the result is `rejected`: explain the reason in plain words (see policy.md) and offer
   the alternatives listed in the result. Do not try another reason to get around a rejection.
4. If the result is `needs_confirmation`: tell the customer, in this order,
   - the item (name, size, quantity) and the reason,
   - the refund estimate, naming any restocking fee or label fee that was deducted,
   - where the refund goes and how long it takes,
   then ask: "Shall I submit this return?" Do nothing else in that turn.
5. Only when the customer explicitly agrees (in a later message) call `confirm_return` with
   the token. Give them the ticket id and say approval usually takes a few minutes.
6. For "is it approved yet?" call `check_return(ticket_id)`. When approved, give the label
   link and the drop-off instructions from the result. Never paste file contents; the link is enough.

Exchanges are not supported: offer a return plus a new order.
