# =============================================================================
# UNIFIED SENTIMENT ANALYSIS BENCHMARK
# (VADER + TRANSFORMERS: DistilBERT, RoBERTa)
# =============================================================================

import io
import ssl
import tarfile
import urllib.request
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error

# VADER
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Transformers
from transformers import pipeline
import torch

# =============================================================================
# CONFIG
# =============================================================================

VADER_TARBALL_URL = "https://files.pythonhosted.org/packages/77/8c/4a48c10a50f750ae565e341e697d74a38075a3e43ff0df6f1ab72e186902/vaderSentiment-3.3.2.tar.gz"

CORPORA = {
    "Amazon": "hutto_ICWSM_2014/amazonReviewSnippets_GroundTruth.txt",
    "Tweets": "hutto_ICWSM_2014/tweets_GroundTruth.txt",
    "NYT": "hutto_ICWSM_2014/nytEditorialSnippets_GroundTruth.txt",
    "Movies": "hutto_ICWSM_2014/movieReviewSnippets_GroundTruth.txt",
}

HUMAN_POS_THRESH = 0.5
HUMAN_NEG_THRESH = -0.5

# =============================================================================
# DATA LOADING
# =============================================================================

def fetch_data():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    print("[1] Downloading dataset...")
    resp = urllib.request.urlopen(VADER_TARBALL_URL, context=ctx)
    outer = tarfile.open(fileobj=io.BytesIO(resp.read()), mode="r:gz")

    inner_bytes = outer.extractfile(
        "vaderSentiment-3.3.2/additional_resources/hutto_ICWSM_2014.tar.gz"
    ).read()

    tf = tarfile.open(fileobj=io.BytesIO(inner_bytes), mode="r:gz")

    dfs = []
    for name, path in CORPORA.items():
        raw = tf.extractfile(path).read().decode("utf-8", errors="replace")

        rows = []
        for line in raw.splitlines():
            parts = line.split("\t", 2)
            if len(parts) == 3:
                rows.append({
                    "id": parts[0],
                    "human_score": float(parts[1]),
                    "text": parts[2],
                    "corpus": name
                })

        dfs.append(pd.DataFrame(rows))

    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df)} samples\n")
    return df

# =============================================================================
# LABEL FUNCTIONS
# =============================================================================

def human_label(score):
    if score >= HUMAN_POS_THRESH:
        return "POSITIVE"
    elif score <= HUMAN_NEG_THRESH:
        return "NEGATIVE"
    return "NEUTRAL"

def vader_label(comp):
    if comp >= 0.05:
        return "POSITIVE"
    elif comp <= -0.05:
        return "NEGATIVE"
    return "NEUTRAL"

def map_transformer_label(label):
    label = label.upper()
    if "POS" in label:
        return "POSITIVE", 1
    elif "NEG" in label:
        return "NEGATIVE", -1
    return "NEUTRAL", 0

# =============================================================================
# VADER
# =============================================================================

def run_vader(df):
    print("[2] Running VADER...")
    sia = SentimentIntensityAnalyzer()

    comps = []
    labels = []

    for text in df["text"]:
        s = sia.polarity_scores(str(text))
        comps.append(s["compound"])
        labels.append(vader_label(s["compound"]))

    df["vader_score"] = np.array(comps) * 4
    df["vader_label"] = labels
    return df

# =============================================================================
# TRANSFORMERS
# =============================================================================

def load_models():
    print("[3] Loading transformer models...")

    device = 0 if torch.cuda.is_available() else -1

    models = {
        "distilbert": pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=device
        ),
        "roberta": pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment",
            device=device
        )
    }

    return models

def run_transformers(df, models):
    print("[4] Running transformers...")

    for name, model in models.items():
        preds = []
        scores = []

        texts = df["text"].tolist()

        for i in range(0, len(texts), 32):
            batch = texts[i:i+32]
            outputs = model(batch, truncation=True)

            for out in outputs:
                lbl, sc = map_transformer_label(out["label"])
                preds.append(lbl)
                scores.append(sc)

        df[f"{name}_label"] = preds
        df[f"{name}_score"] = np.array(scores) * 4

    return df

# =============================================================================
# EVALUATION
# =============================================================================

def evaluate(df, model_names):
    print("[5] Evaluation\n")

    for model in model_names:
        print(f"--- {model.upper()} ---")

        y_true = df["human_label"]
        y_pred = df[f"{model}_label"]

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro")

        mae = mean_absolute_error(df["human_score"], df[f"{model}_score"])
        rmse = mean_squared_error(df["human_score"], df[f"{model}_score"]) ** 0.5

        print(f"Accuracy : {acc:.4f}")
        print(f"Macro-F1 : {f1:.4f}")
        print(f"MAE      : {mae:.4f}")
        print(f"RMSE     : {rmse:.4f}\n")

# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_comparison(df):
    print("[6] Plotting...")

    models = ["vader", "distilbert", "roberta"]
    accs = []
    f1s = []

    for m in models:
        accs.append(accuracy_score(df["human_label"], df[f"{m}_label"]))
        f1s.append(f1_score(df["human_label"], df[f"{m}_label"], average="macro"))

    x = np.arange(len(models))

    plt.figure(figsize=(8,5))
    plt.bar(x-0.2, accs, width=0.4, label="Accuracy")
    plt.bar(x+0.2, f1s, width=0.4, label="Macro-F1")

    plt.xticks(x, models)
    plt.title("Model Comparison")
    plt.legend()
    plt.savefig("comparison.png")
    plt.close()

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n=== SENTIMENT BENCHMARK ===\n")

    df = fetch_data()

    df["human_label"] = df["human_score"].apply(human_label)

    df = run_vader(df)

    models = load_models()
    df = run_transformers(df, models)

    evaluate(df, ["vader", "distilbert", "roberta"])

    plot_comparison(df)

    df.to_csv("results.csv", index=False)

    print("\n✅ DONE")
    print("Saved: results.csv, comparison.png\n")


if __name__ == "__main__":
    main()