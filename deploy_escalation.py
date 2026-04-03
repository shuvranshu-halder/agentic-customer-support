# final_pipeline_clean.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# ============================
# 🔹 1. SIMPLE RAG SYSTEM
# ============================
class SimpleRAG:
    def __init__(self):

        # Embedding model
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

        # Knowledge base (replace later with Flipkart/Amazon policies)
        self.docs = [
            "You can return items within 7 days.",
            "Refunds are processed within 5 business days.",
            "Orders are delivered within 3-5 days.",
            "You can cancel your order before shipment."
        ]

        # Create FAISS index
        embeddings = self.embedder.encode(self.docs)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(np.array(embeddings))

        # LLM generator
        self.generator = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
        self.gen_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

    def retrieve(self, query):
        q_emb = self.embedder.encode([query])
        D, I = self.index.search(np.array(q_emb), k=1)
        return self.docs[I[0][0]]

    def generate(self, query):
        context = self.retrieve(query)
        prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
        inputs = self.gen_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        outputs = self.generator.generate(**inputs, max_length=128, num_beams=4, early_stopping=True)
        response = self.gen_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response, context


# ============================
# 🔹 2. LIGHTWEIGHT EVALUATION (RAGAS-FREE)
# ============================
class SimpleEvaluator:
    def __init__(self):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

    def cosine_sim(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def evaluate(self, query, response, context):

        q_emb = self.embedder.encode(query)
        r_emb = self.embedder.encode(response)
        c_emb = self.embedder.encode(context)

        # Faithfulness ~ similarity between response and context
        faith = self.cosine_sim(r_emb, c_emb)

        # Relevance ~ similarity between query and response
        relevance = self.cosine_sim(q_emb, r_emb)

        # Context usage ~ similarity between query and context
        context_score = self.cosine_sim(q_emb, c_emb)

        return {
            "faithfulness": float(faith),
            "answer_relevancy": float(relevance),
            "context_relevancy": float(context_score)
        }


# ============================
# 🔹 3. ESCALATION MODEL
# ============================
class EscalationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained("roberta-base")
        hidden = self.encoder.config.hidden_size

        self.fc = nn.Linear(hidden + 4, 2)

    def forward(self, input_ids, attention_mask, features):
        out = self.encoder(input_ids=input_ids,
                           attention_mask=attention_mask)

        cls = out.last_hidden_state[:, 0]
        x = torch.cat([cls, features], dim=1)
        return self.fc(x)


# ============================
# 🔹 4. DECISION ENGINE
# ============================
class DecisionEngine:

    def decide(self, features, model_prob):
        faith, rel, ctx, sentiment = features

        # Rule-based override
        if faith < 0.7 or rel < 0.7 or sentiment < -0.5:
            return "ESCALATE", 1.0

        # Model-based
        return ("ESCALATE" if model_prob > 0.5 else "OK"), model_prob


# ============================
# 🔹 5. FULL PIPELINE
# ============================
class FullPipeline:

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.rag = SimpleRAG()
        self.evaluator = SimpleEvaluator()
        self.model = EscalationModel().to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        self.sentiment = SentimentIntensityAnalyzer()
        self.decision = DecisionEngine()

    def run(self, query):

        # ------------------
        # 1. RAG
        # ------------------
        response, context = self.rag.generate(query)

        # ------------------
        # 2. Evaluation
        # ------------------
        scores = self.evaluator.evaluate(query, response, context)

        faith = scores["faithfulness"]
        rel = scores["answer_relevancy"]
        ctx = scores["context_relevancy"]

        # ------------------
        # 3. Sentiment
        # ------------------
        sentiment = self.sentiment.polarity_scores(query)["compound"]

        features = torch.tensor([[faith, rel, ctx, sentiment]]).float().to(self.device)

        # ------------------
        # 4. Model inference
        # ------------------
        text = query + " [SEP] " + response

        inputs = self.tokenizer(text,
                                return_tensors="pt",
                                truncation=True,
                                padding=True).to(self.device)

        with torch.no_grad():
            logits = self.model(inputs["input_ids"],
                                inputs["attention_mask"],
                                features)
            probs = F.softmax(logits, dim=-1)

        model_prob = probs[0][1].item()

        # ------------------
        # 5. Decision
        # ------------------
        decision, confidence = self.decision.decide(
            [faith, rel, ctx, sentiment],
            model_prob
        )

        return {
            "query": query,
            "response": response,
            "faithfulness": faith,
            "relevance": rel,
            "context": ctx,
            "sentiment": sentiment,
            "decision": decision,
            "confidence": confidence
        }


# ============================
# 🔹 RUN
# ============================
if __name__ == "__main__":

    pipeline = FullPipeline()

    query = "My order is not delivered and I am very angry!"

    result = pipeline.run(query)

    print("\n===== FINAL OUTPUT =====")
    print(result)