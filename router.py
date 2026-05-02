import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"   # same GPU as your accuracy_check notebook
os.environ["HOME"] = "/mnt/nas/shuvranshu"
os.environ["HF_HOME"] = "/mnt/nas/shuvranshu/huggingface_cache"
os.environ["TRANSFORMERS_CACHE"] = "/mnt/nas/shuvranshu/huggingface_cache"

import torch
import safetensors
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal, Optional

# ============================================================
# Load Qwen (same loading pattern as your accuracy_check.ipynb)
# ============================================================

SAVE_DIR = "/mnt/nas/shuvranshu/finetune"
base_model_path = "Qwen/Qwen2.5-3B-Instruct"
lora_adapter_path = f"{SAVE_DIR}/qwen_lora_adapter"

tokenizer = AutoTokenizer.from_pretrained(base_model_path)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

peft_config = PeftConfig.from_pretrained(lora_adapter_path)
qwen_model = PeftModel(base_model, peft_config)
adapter_weights = safetensors.torch.load_file(f"{lora_adapter_path}/model.safetensors")
qwen_model.load_state_dict(adapter_weights, strict=False)
qwen_model.to("cuda")
qwen_model.eval()

print("✅ Qwen classifier loaded")

# ============================================================
# Valid categories (must match your CATEGORY_MAP output exactly)
# ============================================================

VALID_CATEGORIES = {"billing", "account", "technical", "shipping", "order", "complaint", "other"}

SYSTEM_PROMPT = """You are a customer support query classifier.
Classify the user query into exactly one of these categories:
[billing, account, technical, shipping, order, complaint, other]
Respond with ONLY the category name, nothing else."""

# ============================================================
# Qwen inference (same decode logic as accuracy_check.ipynb)
# ============================================================

def qwen_classify(query: str) -> tuple[str, float]:
    """
    Returns (intent, confidence).
    Confidence is estimated via max token probability of the first generated token.
    """
    input_text = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{query}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to("cuda")

    with torch.no_grad():
        # Get logits for confidence estimation
        outputs = qwen_model(input_ids)
        logits = outputs.logits[:, -1, :]  # last token logits
        probs = torch.softmax(logits, dim=-1)
        confidence = probs.max().item()

        # Generate category
        generated = qwen_model.generate(input_ids, max_new_tokens=5)

    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True).strip().lower()
    predicted = generated_text.split("assistant")[-1].strip()

    # Validate — if Qwen hallucinates an unknown label, fall back to "other"
    if predicted not in VALID_CATEGORIES:
        predicted = "other"

    return predicted, confidence


# ============================================================
# State definition
# ============================================================

class SupportState(TypedDict):
    user_input: str
    intent: str
    confidence: float
    sentiment: str          # filled by BERT node
    escalate: bool
    response: str
    fallback_reason: Optional[str]   # why we fell back (for logging)


# ============================================================
# Node 1 — Classify with Qwen
# ============================================================

def classify_node(state: SupportState) -> SupportState:
    intent, confidence = qwen_classify(state["user_input"])
    return {**state, "intent": intent, "confidence": confidence}


# ============================================================
# Router — decides next node based on intent + confidence
# ============================================================

CONFIDENCE_THRESHOLD = 0.60   # below this → treat as low-confidence → RAG fallback

def route_intent(state: SupportState) -> Literal["rag", "tool_agent", "escalate", "reject", "low_confidence"]:
    intent = state["intent"]
    confidence = state["confidence"]

    # Low confidence on any intent → safest fallback is RAG
    if confidence < CONFIDENCE_THRESHOLD:
        return "low_confidence"

    if intent in ["faq", "complaint"]:
        return "rag"
    elif intent in ["order", "shipping"]:
        return "tool_agent"
    elif intent in ["billing", "account"]:
        # Billing/account can need both tool + RAG depending on sub-query
        # Route to tool_agent which internally decides
        return "tool_agent"
    elif intent == "technical":
        return "rag"
    elif intent == "other":
        return "reject"
    else:
        # Should never reach here after VALID_CATEGORIES guard, but safety net
        return "low_confidence"


# ============================================================
# Node 2a — RAG Llama node (plug in your existing Llama RAG here)
# ============================================================

def rag_node(state: SupportState) -> SupportState:
    # ---- Replace this block with your Llama RAG call ----
    # response = llama_rag(state["user_input"])
    response = f"[RAG] Answering: {state['user_input']}"
    # ------------------------------------------------------
    return {**state, "response": response}


# ============================================================
# Node 2b — Tool Agent node (order/billing/account actions)
# ============================================================

TOOL_DISPATCH = {
    "order"    : "check_order_status",
    "shipping" : "track_shipment",
    "billing"  : "billing_action",    # refund / payment / cancellation
    "account"  : "account_action",    # reset password / update info
}

def tool_agent_node(state: SupportState) -> SupportState:
    tool = TOOL_DISPATCH.get(state["intent"], "generic_tool")

    # ---- Replace with your actual tool calls ----
    # response = call_tool(tool, state["user_input"])
    response = f"[TOOL:{tool}] Handling: {state['user_input']}"
    # ---------------------------------------------
    return {**state, "response": response}


# ============================================================
# Node 3 — Low confidence fallback
# ============================================================

def low_confidence_node(state: SupportState) -> SupportState:
    """
    When Qwen isn't confident, fall back to RAG.
    Log the reason so you can use it to collect hard examples for retraining.
    """
    # ---- Replace with your Llama RAG call ----
    response = f"[LOW_CONF_RAG] Answering: {state['user_input']}"
    # ------------------------------------------
    return {
        **state,
        "response": response,
        "fallback_reason": f"low_confidence:{state['confidence']:.2f}|raw_intent:{state['intent']}"
    }


# ============================================================
# Node 4 — Escalate to human agent
# ============================================================

def escalate_node(state: SupportState) -> SupportState:
    return {
        **state,
        "response": "I'm connecting you to a human support agent. Please hold on.",
        "escalate": True
    }


# ============================================================
# Node 5 — Out of scope rejection
# ============================================================

def reject_node(state: SupportState) -> SupportState:
    return {
        **state,
        "response": "I can only assist with billing, orders, shipping, account, and technical support queries."
    }


# ============================================================
# Node 6 — BERT sentiment check (runs after every response)
# ============================================================

def bert_sentiment_node(state: SupportState) -> SupportState:
    """
    Plug in your BERT sentiment model here.
    If sentiment is very negative, override to escalate even if intent wasn't escalation.
    """
    # ---- Replace with your BERT model call ----
    # sentiment = bert_model(state["user_input"])  # returns "positive"/"neutral"/"negative"/"very_negative"
    sentiment = "neutral"   # placeholder
    # -------------------------------------------

    if sentiment == "very_negative" and not state["escalate"]:
        return {
            **state,
            "sentiment": sentiment,
            "escalate": True,
            "response": (
                "I can see you're frustrated and I'm sorry for that experience. "
                "Let me connect you to a senior support agent right away."
            )
        }

    return {**state, "sentiment": sentiment}


# ============================================================
# Node 7 — Final escalation check after sentiment
# (routes to human handoff if escalate=True after sentiment node)
# ============================================================

def post_sentiment_route(state: SupportState) -> Literal["escalate", END]:
    if state["escalate"]:
        return "escalate"
    return END


# ============================================================
# Build the LangGraph
# ============================================================

graph = StateGraph(SupportState)

graph.add_node("classify",        classify_node)
graph.add_node("rag",             rag_node)
graph.add_node("tool_agent",      tool_agent_node)
graph.add_node("low_confidence",  low_confidence_node)
graph.add_node("escalate",        escalate_node)
graph.add_node("reject",          reject_node)
graph.add_node("sentiment_check", bert_sentiment_node)

# Entry
graph.set_entry_point("classify")

# Classify → conditional routing
graph.add_conditional_edges("classify", route_intent, {
    "rag"            : "rag",
    "tool_agent"     : "tool_agent",
    "escalate"       : "escalate",
    "reject"         : "reject",
    "low_confidence" : "low_confidence",
})

# All response-generating nodes → sentiment check
graph.add_edge("rag",            "sentiment_check")
graph.add_edge("tool_agent",     "sentiment_check")
graph.add_edge("low_confidence", "sentiment_check")

# Sentiment check → escalate if needed, else END
graph.add_conditional_edges("sentiment_check", post_sentiment_route, {
    "escalate" : "escalate",
    END        : END,
})

# Terminal nodes
graph.add_edge("escalate", END)
graph.add_edge("reject",   END)

app = graph.compile()


# ============================================================
# Helper: run a query and print result
# ============================================================

def run_support_query(user_input: str) -> dict:
    initial_state: SupportState = {
        "user_input"     : user_input,
        "intent"         : "",
        "confidence"     : 0.0,
        "sentiment"      : "",
        "escalate"       : False,
        "response"       : "",
        "fallback_reason": None,
    }
    result = app.invoke(initial_state)
    print(f"\nQuery      : {result['user_input']}")
    print(f"Intent     : {result['intent']}  (conf={result['confidence']:.2f})")
    print(f"Sentiment  : {result['sentiment']}")
    print(f"Escalate   : {result['escalate']}")
    if result["fallback_reason"]:
        print(f"Fallback   : {result['fallback_reason']}")
    print(f"Response   : {result['response']}")
    return result


# ============================================================
# Quick test
# ============================================================

if __name__ == "__main__":
    test_queries = [
        "I want a refund for my order #1234",          # billing → tool_agent
        "My internet is not working since yesterday",  # technical → rag
        "Where is my package?",                        # shipping → tool_agent
        "I cannot log into my account",                # account → tool_agent
        "This is absolutely terrible service!!!",      # complaint → rag → BERT escalate
        "I want to speak to a manager",                # other → reject (Qwen maps it to other)
        "What's the weather today?",                   # out of scope → reject
    ]

    for q in test_queries:
        run_support_query(q)
