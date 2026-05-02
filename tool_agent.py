# tool_agent.py

import json
import re
from typing import Optional
from rag import llama_generate   # reuse your existing Llama from rag.py

# ============================================================
# Fake DB (replace with your real DB calls later)
# ============================================================

ORDERS_DB = {
    "ORD001": {"product_id": "P100", "product": "Wireless Headphones",
               "status": "shipped", "customer": "John", "amount": 59.99},
    "ORD002": {"product_id": "P200", "product": "USB Hub",
               "status": "processing", "customer": "Sara", "amount": 29.99},
    "ORD003": {"product_id": "P300", "product": "Laptop Stand",
               "status": "delivered", "customer": "Mike", "amount": 45.00},
}

SHIPMENTS_DB = {
    "ORD001": {"carrier": "FedEx", "tracking_id": "FX123456",
               "estimated_delivery": "2025-05-05", "current_location": "Mumbai Hub"},
    "ORD002": {"carrier": "BlueDart", "tracking_id": "BD789012",
               "estimated_delivery": "2025-05-07", "current_location": "Warehouse"},
}

PAYMENTS_DB = {
    "ORD001": {"status": "paid", "method": "Credit Card", "date": "2025-04-28"},
    "ORD002": {"status": "paid", "method": "UPI",         "date": "2025-04-30"},
    "ORD003": {"status": "refunded",  "method": "Debit Card","date": "2025-04-20"},
}

USERS_DB = {
    "john@email.com": {"name": "John", "plan": "Premium",
                       "joined": "2023-01-15", "open_tickets": 2},
}

# ============================================================
# Tool functions — each hits the "DB" and returns raw data
# ============================================================

def check_order_status(order_id: str = None, product_id: str = None) -> dict:
    """Look up by order_id OR product_id."""
    if order_id and order_id in ORDERS_DB:
        return {"found": True, "order_id": order_id, **ORDERS_DB[order_id]}

    if product_id:
        for oid, data in ORDERS_DB.items():
            if data["product_id"] == product_id:
                return {"found": True, "order_id": oid, **data}

    return {"found": False, "error": "Order not found. Please check your order ID or product ID."}


def track_shipment(order_id: str = None, product_id: str = None) -> dict:
    """Track shipment — needs order_id. If product_id given, resolve first."""
    if not order_id and product_id:
        order = check_order_status(product_id=product_id)
        if order["found"]:
            order_id = order["order_id"]

    if order_id and order_id in SHIPMENTS_DB:
        order_info = ORDERS_DB.get(order_id, {})
        return {"found": True, "order_id": order_id,
                "product": order_info.get("product", "Unknown"),
                **SHIPMENTS_DB[order_id]}

    # Order exists but not shipped yet
    if order_id and order_id in ORDERS_DB:
        status = ORDERS_DB[order_id]["status"]
        return {"found": True, "order_id": order_id,
                "message": f"Order is currently '{status}' — tracking not available yet."}

    return {"found": False, "error": "Shipment not found."}


def process_refund(order_id: str = None, product_id: str = None) -> dict:
    """Check eligibility and process refund."""
    if not order_id and product_id:
        order = check_order_status(product_id=product_id)
        if order["found"]:
            order_id = order["order_id"]

    if not order_id or order_id not in ORDERS_DB:
        return {"found": False, "error": "Order not found for refund."}

    order = ORDERS_DB[order_id]
    payment = PAYMENTS_DB.get(order_id, {})

    # Business rules
    if payment.get("status") == "refunded":
        return {"found": True, "eligible": False,
                "reason": "This order has already been refunded."}

    if order["status"] == "processing":
        return {"found": True, "eligible": True,
                "action": "cancel_and_refund",
                "amount": order["amount"],
                "message": "Order is still processing — we can cancel and refund immediately."}

    if order["status"] == "delivered":
        return {"found": True, "eligible": True,
                "action": "return_refund",
                "amount": order["amount"],
                "message": "Delivered order — return shipping label will be emailed to you."}

    return {"found": True, "eligible": False,
            "reason": f"Cannot refund order with status '{order['status']}'."}


def check_payment_status(order_id: str = None, product_id: str = None) -> dict:
    if not order_id and product_id:
        order = check_order_status(product_id=product_id)
        if order["found"]:
            order_id = order["order_id"]

    if order_id and order_id in PAYMENTS_DB:
        return {"found": True, "order_id": order_id, **PAYMENTS_DB[order_id]}

    return {"found": False, "error": "Payment record not found."}


def cancel_order(order_id: str = None, product_id: str = None) -> dict:
    if not order_id and product_id:
        order = check_order_status(product_id=product_id)
        if order["found"]:
            order_id = order["order_id"]

    if not order_id or order_id not in ORDERS_DB:
        return {"found": False, "error": "Order not found."}

    status = ORDERS_DB[order_id]["status"]
    if status == "processing":
        return {"found": True, "cancelled": True,
                "message": "Order cancelled successfully. Refund will be processed in 3-5 business days."}
    if status == "shipped":
        return {"found": True, "cancelled": False,
                "reason": "Order already shipped. Please use the refund option instead."}
    if status == "delivered":
        return {"found": True, "cancelled": False,
                "reason": "Order already delivered. Please use the refund option instead."}

    return {"found": False, "error": f"Cannot cancel order with status '{status}'."}


def get_account_info(email: str = None) -> dict:
    if email and email in USERS_DB:
        return {"found": True, **USERS_DB[email]}
    return {"found": False,
            "error": "Account not found. Please verify your email address."}


# ============================================================
# Tool registry — Llama picks from this
# ============================================================

TOOLS = {
    "check_order_status" : check_order_status,
    "track_shipment"     : track_shipment,
    "process_refund"     : process_refund,
    "check_payment_status": check_payment_status,
    "cancel_order"       : cancel_order,
    "get_account_info"   : get_account_info,
}

TOOL_DESCRIPTIONS = """
Available tools:
- check_order_status(order_id, product_id): Get order status and details
- track_shipment(order_id, product_id): Get shipment tracking info
- process_refund(order_id, product_id): Check refund eligibility and process it
- check_payment_status(order_id, product_id): Get payment status
- cancel_order(order_id, product_id): Cancel a processing order
- get_account_info(email): Get user account details
"""

# ============================================================
# Step 1 — Llama picks the right tool and extracts params
# ============================================================

TOOL_SELECTION_PROMPT = """You are a customer support tool selector.

{tool_descriptions}

Given this customer query, respond ONLY with a JSON object like:
{{"tool": "tool_name", "params": {{"order_id": "ORD001"}}}}

Rules:
- Use null for params you cannot extract from the query
- For billing queries: check keywords — "refund"->process_refund, "payment"->check_payment_status, "cancel"->cancel_order
- If the query mentions a product ID (like P100, P200), use product_id param
- If the query mentions an order ID (like ORD001), use order_id param
- For account queries, extract email if present
- Respond with ONLY the JSON, no explanation

Query: {query}
"""

def llama_select_tool(query: str) -> tuple[str, dict]:
    """Ask Llama which tool to call and what params to pass."""
    prompt = TOOL_SELECTION_PROMPT.format(
        tool_descriptions=TOOL_DESCRIPTIONS,
        query=query
    )
    raw = llama_generate(prompt)   # your existing function from rag.py

    # Parse JSON from Llama output
    try:
        # Extract JSON block if Llama adds extra text
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            parsed = json.loads(match.group())
            tool_name = parsed.get("tool", "")
            params = parsed.get("params", {})
            # Remove null params
            params = {k: v for k, v in params.items() if v is not None}
            return tool_name, params
    except (json.JSONDecodeError, AttributeError):
        pass

    return None, {}


# ============================================================
# Step 2 — Llama formats the DB result into a human response
# ============================================================

RESPONSE_FORMAT_PROMPT = """You are a helpful customer support agent.

The customer asked: {query}

You looked up the database and got this result:
{db_result}

Write a friendly, concise response to the customer based on this data.
Do NOT make up any information not in the result.
If the result shows an error or not found, apologize and ask for correct details.
"""

def llama_format_response(query: str, db_result: dict) -> str:
    prompt = RESPONSE_FORMAT_PROMPT.format(
        query=query,
        db_result=json.dumps(db_result, indent=2)
    )
    return llama_generate(prompt)   # your existing function from rag.py


# ============================================================
# Main tool agent entry point (called from router.py)
# ============================================================

def tool_agent(query: str, intent: str) -> str:
    """
    Full tool agent pipeline:
    1. Llama selects tool + extracts params from query
    2. Tool hits the DB
    3. Llama formats DB result into a human-readable response
    """

    # Step 1: Tool selection
    tool_name, params = llama_select_tool(query)

    # Edge case: Llama picked an unknown tool or failed to parse
    if not tool_name or tool_name not in TOOLS:
        # Fallback: use intent to pick the most likely tool
        intent_fallback = {
            "order"   : "check_order_status",
            "shipping": "track_shipment",
            "billing" : "check_payment_status",
            "account" : "get_account_info",
        }
        tool_name = intent_fallback.get(intent, "check_order_status")
        params = {}

    # Edge case: Llama extracted no params — ask user for more info
    if not params:
        return (
            f"To help you with your {intent} query, could you please provide "
            f"your order ID (e.g. ORD001) or product ID (e.g. P100)?"
        )

    # Step 2: Hit the DB
    tool_fn = TOOLS[tool_name]
    db_result = tool_fn(**params)

    # Edge case: DB returned not found
    if not db_result.get("found", False):
        error_msg = db_result.get("error", "No details found.")
        return (
            f"I'm sorry, I couldn't find the details. {error_msg} "
            f"Please double-check your order ID or product ID and try again."
        )

    # Step 3: Format with Llama
    return llama_format_response(query, db_result)