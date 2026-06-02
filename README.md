# Agentic LLM-based Automated Customer Support Resolution and Escalation System
---

## Overview

A production-oriented agentic customer support system that combines intent classification,
Retrieval-Augmented Generation (RAG), and sentiment-aware escalation to automate
query resolution — and intelligently hand off to humans when needed.

---

## Architecture

![workflow](workflow.png)

---

## Modules

### 1. Intent Classification
- Fine-tuned **Qwen-2.5-3B-Instruct** with **LoRA** (rank=16, α=32) on the
  [Bitext Customer Support Dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)
- Labels: `billing`, `account`, `shipping`, `order`, `technical`, `complaint`, `other`
- Routing logic:
  - `billing/account/shipping/order` → Tool Agent
  - `technical/complaint` → RAG Pipeline
  - confidence < 0.6 → RAG fallback
- **Result:** 65% validation accuracy (+23pp over zero-shot baseline)

### 2. RAG Pipeline
- Policy PDFs chunked into 512-token overlapping windows (64-token overlap)
- Embedded via sentence-transformers, stored in **FAISS** vector DB
- Top-5 chunks retrieved at inference via cosine similarity
- **Llama 3.1-8B-Instruct** generates the final response

### 3. Hallucination Mitigation
- A **Knowledge Graph (KG)** extracts structured entities and relationships from
  retrieved chunks, grounding the LLM response in verified domain facts

### 4. Reflexion Loop
An actor–evaluator–refiner loop iteratively improves response quality:
User Query → Retriever → RAG LLM → Evaluator LLM
↑                        │
└── Re-retrieve / Refine ┘
- **PASS** if Context Relevance ≥ 0.8 AND Faithfulness ≥ 0.8
- Otherwise: re-retrieves with a refined query or refines the answer

### 5. Sentiment-Aware Escalation
VADER is used for real-time sentiment scoring. Three signals are combined via
a **weighted harmonic mean** into a health score Ψ:
Ψ = CSO / (wc·SO + ws·CO + wo·CS)
Where:
- **C** = Model confidence (softmax logits)
- **S** = VADER sentiment (normalized)
- **O** = Energy-based OOD score

If Ψ < threshold τ → **ESCALATE TO HUMAN**, else → **PROCEED AUTONOMOUSLY**

---

## Results

### RAG Evaluation (RAGAS on Bitext dataset, 200 samples)

| Metric            | Ours  | Naive RAG |
|-------------------|-------|-----------|
| Context Recall    | 0.839 | 0.712     |
| Faithfulness      | 0.713 | 0.584     |
| Answer Relevancy  | 0.821 | 0.734     |
| Context Precision | 0.789 | 0.655     |

### Sentiment Model Comparison

| Model      | Accuracy | Macro-F1 | Chosen? |
|------------|----------|----------|---------|
| VADER      | ~0.60    | High     | ✅ Yes  |
| DistilBERT | ~0.70    | Medium   |         |
| RoBERTa    | ~0.15    | Low      |         |

VADER selected for its high F1 and lightweight real-time performance.

---

## Tech Stack

| Component        | Technology                        |
|------------------|-----------------------------------|
| Intent Classifier| Qwen-2.5-3B-Instruct + LoRA (PEFT)|
| RAG Generator    | Llama 3.1-8B-Instruct             |
| Vector DB        | FAISS                             |
| Embeddings       | Sentence Transformers             |
| Sentiment        | VADER                             |
| Evaluation       | RAGAS + GPT-4.1                   |
| Evaluator Agent  | GPT-4o-mini                       |

---

## Setup

```bash
git clone https://github.com/shuvranshu-halder/agentic-customer-support
cd agentic-customer-support
pip install -r requirements.txt
```

---

## References

- [LoRA](https://arxiv.org/abs/2106.09685) · [Reflexion](https://arxiv.org/abs/2303.11366) · [RAGAS](https://github.com/explodinggradients/ragas) · [VADER](https://ojs.aaai.org/index.php/ICWSM/article/view/14550) · [Energy-based OOD](https://arxiv.org/abs/2010.03759)
