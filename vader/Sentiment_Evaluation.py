"""
=============================================================================
 VADER Sentiment Analysis — Customer Support Dataset
=============================================================================

 Dataset : Amazon Product Review Snippets
           Source  : VADER ICWSM-2014 research corpus (C.J. Hutto & E. Gilbert)
           Distributed with vaderSentiment PyPI package (MIT License)
           URL     : https://pypi.org/project/vaderSentiment/

           • 3,708 real customer sentences from 300 Amazon product reviews
           • Each sentence carries a mean human sentiment rating [-4, +4]
             averaged over 10 independent Mechanical Turk raters
           • Reviews cover consumer electronics products and authentically
             contain customer-service language: complaints, returns, refunds,
             product quality, support responsiveness, purchase recommendations

 Pipeline :
   1. Download & parse dataset
   2. Rule-based customer-support intent tagging
   3. VADER scoring (neg / neu / pos / compound)
   4. 3-class label derivation  (POSITIVE / NEUTRAL / NEGATIVE)
   5. Escalation-risk flagging
   6. Classification & regression evaluation vs human ground truth
   7. 8-panel analytics dashboard (PNG)
   8. Export full results to CSV

 VADER Compound Thresholds (standard):
   compound >=  0.05  → POSITIVE
   compound <= -0.05  → NEGATIVE
   otherwise          → NEUTRAL
=============================================================================
"""

# ── stdlib ────────────────────────────────────────────────────────────────
import io, re, ssl, tarfile, urllib.request, warnings
warnings.filterwarnings("ignore")

# ── third-party ───────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score,
    mean_absolute_error, mean_squared_error,
)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ── constants ─────────────────────────────────────────────────────────────
VADER_TARBALL = (
    "https://files.pythonhosted.org/packages/77/8c/"
    "4a48c10a50f750ae565e341e697d74a38075a3e43ff0df6f1ab72e186902/"
    "vaderSentiment-3.3.2.tar.gz"
)
AMAZON_PATH   = "hutto_ICWSM_2014/amazonReviewSnippets_GroundTruth.txt"

VADER_POS =  0.05
VADER_NEG = -0.05
HUMAN_POS =  0.5       # human score > 0.5  → POSITIVE
HUMAN_NEG = -0.5       # human score < -0.5 → NEGATIVE

# ── Customer-support intent taxonomy ─────────────────────────────────────
INTENTS = {
    "complaint":       r"(bad|poor|terrible|awful|horrible|worst|sucks|broken|defective|"
                       r"disappointed|useless|garbage|junk|waste|problem|issue|fault|"
                       r"not work|doesn.t work|didn.t work|won.t work)",
    "return_refund":   r"(return|refund|money back|send back|exchange|warranty|replace)",
    "support_contact": r"(customer service|tech support|customer support|contact|call|"
                       r"phone|email|response|responsive|answer|help desk|helpdesk)",
    "shipping":        r"(ship|deliver|delivery|arrived|package|order|receiv|transit|"
                       r"tracking|late|delay|fast|quick)",
    "product_quality": r"(quality|build|material|durability|durable|cheap|solid|sturdy|"
                       r"flimsy|break|broke|scratch|crack|damage|defect)",
    "price_value":     r"(price|cost|value|worth|expensive|cheap|afford|budget|deal|"
                       r"bargain|money|dollar|pay|paid)",
    "recommendation":  r"(recommend|suggest|buy|purchase|not recommend|avoid|don.t buy|"
                       r"stay away|go for|worth buying|great buy)",
    "neutral_inquiry": r"(question|wonder|ask|know|information|detail|spec|how to|"
                       r"feature|function|capability)",
}

ESCALATION_PATTERNS = re.compile(
    r"(never again|lawsuit|sue|legal|fraud|scam|disgusting|unacceptable|"
     r"furious|outraged|demand|refund immediately|worst ever|terrible service|"
     r"no response|ignored|lied|false advertising|report|bbb|attorney)",
    re.I
)

PALETTE = {"POSITIVE": "#27ae60", "NEUTRAL": "#95a5a6", "NEGATIVE": "#e74c3c"}
INTENT_COLORS = sns.color_palette("tab10", len(INTENTS))

sns.set_theme(style="whitegrid", font_scale=1.05)


# ══════════════════════════════════════════════════════════════════════════
#  1. DATA LOADING
# ══════════════════════════════════════════════════════════════════════════

def ssl_ctx():
    c = ssl.create_default_context()
    c.check_hostname = False
    c.verify_mode = ssl.CERT_NONE
    return c


def load_amazon_reviews() -> pd.DataFrame:
    """Download real Amazon customer review sentences with human sentiment scores."""
    print("[1/6] Downloading Amazon Customer Review dataset …")
    resp      = urllib.request.urlopen(VADER_TARBALL, context=ssl_ctx(), timeout=60)
    outer_tf  = tarfile.open(fileobj=io.BytesIO(resp.read()), mode="r:gz")
    inner_raw = outer_tf.extractfile(
        "vaderSentiment-3.3.2/additional_resources/hutto_ICWSM_2014.tar.gz"
    ).read()
    inner_tf  = tarfile.open(fileobj=io.BytesIO(inner_raw), mode="r:gz")
    raw_text  = inner_tf.extractfile(AMAZON_PATH).read().decode("utf-8", errors="replace")

    records = []
    for line in raw_text.splitlines():
        parts = line.strip().split("\t", 2)
        if len(parts) != 3:
            continue
        sent_id, score_str, text = parts
        try:
            review_id = int(sent_id.strip().split("_")[0])
            records.append({
                "sentence_id": sent_id.strip(),
                "review_id":   review_id,
                "human_score": float(score_str.strip()),
                "text":        text.strip(),
            })
        except ValueError:
            continue

    df = pd.DataFrame(records)
    print(f"   ✓ {len(df):,} sentences from {df['review_id'].nunique()} customer reviews\n")
    return df


# ══════════════════════════════════════════════════════════════════════════
#  2. INTENT TAGGING  (rule-based)
# ══════════════════════════════════════════════════════════════════════════

def tag_intents(df: pd.DataFrame) -> pd.DataFrame:
    """Assign customer-support intent categories (can be multi-label)."""
    print("[2/6] Tagging customer-support intents …")

    for intent, pattern in INTENTS.items():
        df[f"intent_{intent}"] = df["text"].str.contains(pattern, case=False, regex=True).astype(int)

    # Primary intent = first match (for display); "other" if none match
    intent_cols = [f"intent_{k}" for k in INTENTS]

    def primary(row):
        for col in intent_cols:
            if row[col]:
                return col.replace("intent_", "")
        return "other"

    df["primary_intent"] = df.apply(primary, axis=1)
    df["intent_count"]   = df[intent_cols].sum(axis=1)
    df["escalation_risk"] = df["text"].str.contains(ESCALATION_PATTERNS).astype(int)

    # Coverage stats
    tagged = (df["intent_count"] > 0).sum()
    print(f"   ✓ {tagged:,} sentences tagged ({tagged/len(df)*100:.1f}%)")
    for intent in INTENTS:
        n = df[f"intent_{intent}"].sum()
        print(f"     {intent:<20} → {n:>4} sentences")
    esc = df["escalation_risk"].sum()
    print(f"   ⚠  escalation_risk       → {esc:>4} sentences\n")
    return df


# ══════════════════════════════════════════════════════════════════════════
#  3. VADER SCORING
# ══════════════════════════════════════════════════════════════════════════

def run_vader(df: pd.DataFrame) -> pd.DataFrame:
    """Score every sentence with VADER and assign sentiment labels."""
    print("[3/6] Running VADER sentiment analysis …")
    sia = SentimentIntensityAnalyzer()

    scores = df["text"].apply(lambda t: pd.Series(sia.polarity_scores(str(t))))
    scores.columns = ["vader_neg", "vader_neu", "vader_pos", "vader_compound"]
    df = pd.concat([df, scores], axis=1)

    df["human_label"] = df["human_score"].apply(
        lambda s: "POSITIVE" if s >= HUMAN_POS else ("NEGATIVE" if s <= HUMAN_NEG else "NEUTRAL")
    )
    df["vader_label"] = df["vader_compound"].apply(
        lambda c: "POSITIVE" if c >= VADER_POS else ("NEGATIVE" if c <= VADER_NEG else "NEUTRAL")
    )
    print(f"   ✓ Scored {len(df):,} sentences\n")
    return df


# ══════════════════════════════════════════════════════════════════════════
#  4. EVALUATION
# ══════════════════════════════════════════════════════════════════════════

LABEL_ORDER = ["POSITIVE", "NEUTRAL", "NEGATIVE"]


def evaluate(df: pd.DataFrame) -> dict:
    """Compute overall + per-intent classification and regression metrics."""
    print("[4/6] Evaluating VADER vs human ground truth …")
    results = {}

    def _m(sub, label):
        if len(sub) == 0:
            return None
        yt, yp = sub["human_label"], sub["vader_label"]
        h,  c  = sub["human_score"], sub["vader_compound"]
        return {
            "label":        label,
            "n":            len(sub),
            "accuracy":     accuracy_score(yt, yp),
            "macro_f1":     f1_score(yt, yp, average="macro",    zero_division=0),
            "weighted_f1":  f1_score(yt, yp, average="weighted", zero_division=0),
            "mae":          mean_absolute_error(h, c * 4),
            "rmse":         mean_squared_error(h, c * 4) ** 0.5,
            "pearson_r":    float(np.corrcoef(h, c)[0, 1]) if len(sub) > 2 else 0.0,
            "report":       classification_report(yt, yp, labels=LABEL_ORDER, zero_division=0),
            "cm":           confusion_matrix(yt, yp, labels=LABEL_ORDER),
            "pos_pct":      (yt == "POSITIVE").mean() * 100,
            "neg_pct":      (yt == "NEGATIVE").mean() * 100,
            "neu_pct":      (yt == "NEUTRAL").mean()  * 100,
        }

    results["overall"] = _m(df, "Overall")
    acc = results["overall"]["accuracy"]
    f1  = results["overall"]["macro_f1"]
    r   = results["overall"]["pearson_r"]
    print(f"   Overall  →  Accuracy={acc:.3f}  Macro-F1={f1:.3f}  Pearson-r={r:.3f}")

    for intent in list(INTENTS.keys()) + ["other"]:
        sub = df[df["primary_intent"] == intent]
        r_m = _m(sub, intent)
        if r_m:
            results[intent] = r_m
            print(f"   {intent:<22} n={r_m['n']:>4}  acc={r_m['accuracy']:.3f}  f1={r_m['macro_f1']:.3f}")

    # Escalation subset
    esc = df[df["escalation_risk"] == 1]
    if len(esc):
        results["escalation"] = _m(esc, "Escalation Risk")
        print(f"   {'escalation_risk':<22} n={results['escalation']['n']:>4}  "
              f"acc={results['escalation']['accuracy']:.3f}  f1={results['escalation']['macro_f1']:.3f}")
    print()
    return results


# ══════════════════════════════════════════════════════════════════════════
#  5. DASHBOARD
# ══════════════════════════════════════════════════════════════════════════

def build_dashboard(df: pd.DataFrame, results: dict, out_path: str):
    print("[5/6] Building analytics dashboard …")

    fig = plt.figure(figsize=(22, 24), facecolor="#f5f6fa")
    fig.suptitle(
        "VADER Sentiment Analysis — Customer Support Dataset\n"
        "Amazon Product Reviews (ICWSM-2014 Real Dataset · 3,708 sentences · 300 reviews)",
        fontsize=16, fontweight="bold", y=0.997, color="#2c3e50"
    )
    gs = gridspec.GridSpec(4, 2, figure=fig,
                           hspace=0.45, wspace=0.30,
                           left=0.06, right=0.97, top=0.96, bottom=0.04)

    # ── Panel 1 : Sentiment distribution (human vs VADER) ─────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    labels = LABEL_ORDER
    human_counts = [df["human_label"].value_counts().get(l, 0) for l in labels]
    vader_counts = [df["vader_label"].value_counts().get(l, 0) for l in labels]
    x = np.arange(3); w = 0.35
    b1 = ax1.bar(x - w/2, human_counts, w, label="Human GT",    color=["#27ae60","#95a5a6","#e74c3c"], alpha=0.85)
    b2 = ax1.bar(x + w/2, vader_counts, w, label="VADER Pred.", color=["#2ecc71","#bdc3c7","#c0392b"], alpha=0.65, hatch="//")
    ax1.set_xticks(x); ax1.set_xticklabels(labels)
    ax1.set_title("Sentiment Distribution: Human GT vs VADER", fontweight="bold")
    ax1.set_ylabel("Sentence Count")
    ax1.legend()
    for bar in list(b1) + list(b2):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
                 str(int(bar.get_height())), ha="center", fontsize=8)

    # ── Panel 2 : Intent distribution with sentiment breakdown ─────────────
    ax2 = fig.add_subplot(gs[0, 1])
    intent_sentiment = df.groupby(["primary_intent", "human_label"]).size().unstack(fill_value=0)
    intent_sentiment = intent_sentiment.reindex(columns=LABEL_ORDER, fill_value=0)
    intent_order = intent_sentiment.sum(axis=1).sort_values(ascending=False).index
    intent_sentiment = intent_sentiment.loc[intent_order]
    intent_sentiment.plot(
        kind="barh", stacked=True, ax=ax2,
        color=[PALETTE[l] for l in LABEL_ORDER], alpha=0.85
    )
    ax2.set_title("Customer Intent Distribution by Sentiment", fontweight="bold")
    ax2.set_xlabel("Sentence Count")
    ax2.set_ylabel("")
    ax2.legend(title="Sentiment", fontsize=8)

    # ── Panel 3 : VADER compound score distribution ────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    for lbl in LABEL_ORDER:
        sub = df[df["human_label"] == lbl]["vader_compound"]
        ax3.hist(sub, bins=40, alpha=0.55, color=PALETTE[lbl], label=f"Human {lbl} (n={len(sub)})", density=True)
    ax3.axvline(VADER_POS, color="#27ae60", ls="--", lw=1.5, label=f"Pos threshold (+{VADER_POS})")
    ax3.axvline(VADER_NEG, color="#e74c3c", ls="--", lw=1.5, label=f"Neg threshold ({VADER_NEG})")
    ax3.set_title("VADER Compound Score by Human Sentiment Label", fontweight="bold")
    ax3.set_xlabel("Compound Score"); ax3.set_ylabel("Density")
    ax3.legend(fontsize=8)

    # ── Panel 4 : Human score vs VADER compound scatter ───────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    for lbl in LABEL_ORDER:
        sub = df[df["human_label"] == lbl]
        ax4.scatter(sub["human_score"], sub["vader_compound"],
                    s=5, alpha=0.3, color=PALETTE[lbl], label=lbl)
    m, b = np.polyfit(df["human_score"], df["vader_compound"], 1)
    xs   = np.linspace(-4, 4, 200)
    r    = results["overall"]["pearson_r"]
    ax4.plot(xs, m*xs+b, color="#2c3e50", lw=2.0, label=f"Regression  r={r:.3f}")
    ax4.set_xlabel("Human Mean Score [-4 → +4]")
    ax4.set_ylabel("VADER Compound [-1 → +1]")
    ax4.set_title("Human Score vs VADER Compound (Customer Reviews)", fontweight="bold")
    ax4.legend(fontsize=8, markerscale=3)

    # ── Panel 5 : Confusion matrix ─────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 0])
    cm      = results["overall"]["cm"]
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues",
                xticklabels=LABEL_ORDER, yticklabels=LABEL_ORDER,
                ax=ax5, linewidths=0.5, cbar_kws={"shrink": 0.8})
    ax5.set_xlabel("VADER Predicted Label")
    ax5.set_ylabel("Human Ground Truth")
    ax5.set_title("Confusion Matrix — Overall (counts, row-normalised colour)", fontweight="bold")

    # ── Panel 6 : Per-intent VADER accuracy ───────────────────────────────
    ax6 = fig.add_subplot(gs[2, 1])
    intent_names = [k for k in results if k not in ("overall", "escalation")]
    intent_accs  = [results[k]["accuracy"]   for k in intent_names]
    intent_f1s   = [results[k]["macro_f1"]   for k in intent_names]
    intent_ns    = [results[k]["n"]           for k in intent_names]
    xi = np.arange(len(intent_names)); w = 0.35
    ax6.bar(xi - w/2, intent_accs, w, color="#3498db", alpha=0.85, label="Accuracy")
    ax6.bar(xi + w/2, intent_f1s,  w, color="#e67e22", alpha=0.85, label="Macro-F1")
    ax6.set_xticks(xi)
    ax6.set_xticklabels([n.replace("_", "\n") for n in intent_names], fontsize=8)
    ax6.set_ylim(0, 1.1)
    ax6.set_title("VADER Performance by Customer Support Intent", fontweight="bold")
    ax6.set_ylabel("Score")
    ax6.legend(fontsize=9)
    for i, (a, f) in enumerate(zip(intent_accs, intent_f1s)):
        ax6.text(i - w/2, a + 0.02, f"{a:.2f}", ha="center", fontsize=7)
        ax6.text(i + w/2, f + 0.02, f"{f:.2f}", ha="center", fontsize=7)

    # ── Panel 7 : Escalation risk analysis ────────────────────────────────
    ax7 = fig.add_subplot(gs[3, 0])
    esc_df  = df[df["escalation_risk"] == 1]
    norm_df = df[df["escalation_risk"] == 0]
    groups  = ["Escalation Risk", "Normal"]
    pos_pct = [
        (esc_df["human_label"]  == "POSITIVE").mean() * 100,
        (norm_df["human_label"] == "POSITIVE").mean() * 100,
    ]
    neu_pct = [
        (esc_df["human_label"]  == "NEUTRAL").mean() * 100,
        (norm_df["human_label"] == "NEUTRAL").mean() * 100,
    ]
    neg_pct = [
        (esc_df["human_label"]  == "NEGATIVE").mean() * 100,
        (norm_df["human_label"] == "NEGATIVE").mean() * 100,
    ]
    xi = np.arange(2)
    ax7.bar(xi, pos_pct, label="POSITIVE", color=PALETTE["POSITIVE"], alpha=0.85)
    ax7.bar(xi, neu_pct, bottom=pos_pct, label="NEUTRAL",  color=PALETTE["NEUTRAL"],  alpha=0.85)
    ax7.bar(xi, neg_pct, bottom=[p+n for p, n in zip(pos_pct, neu_pct)],
            label="NEGATIVE", color=PALETTE["NEGATIVE"], alpha=0.85)
    ax7.set_xticks(xi)
    ax7.set_xticklabels([f"{g}\n(n={len(esc_df) if i==0 else len(norm_df)})"
                         for i, g in enumerate(groups)])
    ax7.set_title("Escalation Risk vs Normal: Sentiment Breakdown", fontweight="bold")
    ax7.set_ylabel("% of sentences")
    ax7.legend(fontsize=9)

    # ── Panel 8 : Review-level aggregate heatmap ──────────────────────────
    ax8 = fig.add_subplot(gs[3, 1])
    rev_agg = df.groupby("review_id").agg(
        mean_human  = ("human_score", "mean"),
        mean_vader  = ("vader_compound", "mean"),
        n_sentences = ("text", "count"),
        n_negative  = ("vader_label", lambda x: (x == "NEGATIVE").sum()),
        n_escalation= ("escalation_risk", "sum"),
    ).reset_index()
    rev_agg["neg_ratio"] = rev_agg["n_negative"] / rev_agg["n_sentences"]

    # Scatter: review-level human vs VADER means
    sc = ax8.scatter(
        rev_agg["mean_human"], rev_agg["mean_vader"],
        c=rev_agg["neg_ratio"], cmap="RdYlGn_r",
        s=rev_agg["n_sentences"] * 3, alpha=0.7, vmin=0, vmax=1
    )
    plt.colorbar(sc, ax=ax8, label="Negative sentence ratio")
    m2, b2 = np.polyfit(rev_agg["mean_human"], rev_agg["mean_vader"], 1)
    xs2 = np.linspace(rev_agg["mean_human"].min(), rev_agg["mean_human"].max(), 100)
    r2  = float(np.corrcoef(rev_agg["mean_human"], rev_agg["mean_vader"])[0, 1])
    ax8.plot(xs2, m2*xs2+b2, "k--", lw=1.5, label=f"r={r2:.3f}")
    ax8.set_xlabel("Review Mean Human Score")
    ax8.set_ylabel("Review Mean VADER Compound")
    ax8.set_title(
        f"Review-Level Agreement  (n={len(rev_agg)} reviews)\n"
        "Bubble size = sentence count, colour = negative ratio",
        fontweight="bold"
    )
    ax8.legend(fontsize=9)

    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"   ✓ Dashboard saved → {out_path}\n")


# ══════════════════════════════════════════════════════════════════════════
#  6. REPORT & EXPORT
# ══════════════════════════════════════════════════════════════════════════

def print_report(results: dict):
    print("[6/6] Final Evaluation Report")
    print("=" * 72)

    o = results["overall"]
    print(f"\n{'OVERALL':^72}")
    print("=" * 72)
    print(f"  Sentences   : {o['n']:,}")
    print(f"  Accuracy    : {o['accuracy']:.4f}")
    print(f"  Macro-F1    : {o['macro_f1']:.4f}")
    print(f"  Weighted-F1 : {o['weighted_f1']:.4f}")
    print(f"  MAE         : {o['mae']:.4f}  (scaled to human ±4 range)")
    print(f"  RMSE        : {o['rmse']:.4f}")
    print(f"  Pearson r   : {o['pearson_r']:.4f}")
    print(f"\n  Sentiment Distribution (Human GT):")
    print(f"    POSITIVE  {o['pos_pct']:5.1f}%")
    print(f"    NEUTRAL   {o['neu_pct']:5.1f}%")
    print(f"    NEGATIVE  {o['neg_pct']:5.1f}%")
    print()
    print(o["report"])

    print("-" * 72)
    print("PER-INTENT SUMMARY")
    print("-" * 72)
    header = f"  {'Intent':<22}  {'n':>5}  {'Acc':>6}  {'MacroF1':>8}  {'Pearson':>7}"
    print(header)
    print("  " + "-" * 54)
    for key in list(INTENTS.keys()) + ["other"]:
        if key not in results:
            continue
        r = results[key]
        print(f"  {key:<22}  {r['n']:>5}  {r['accuracy']:>6.3f}  "
              f"{r['macro_f1']:>8.3f}  {r['pearson_r']:>7.3f}")

    if "escalation" in results:
        e = results["escalation"]
        print("-" * 72)
        print(f"  ESCALATION RISK SUBSET  (n={e['n']})")
        print(f"    Accuracy={e['accuracy']:.3f}  Macro-F1={e['macro_f1']:.3f}")
        print(f"    Negative% (human GT): {e['neg_pct']:.1f}%")

    print("=" * 72)


def save_csv(df: pd.DataFrame, path: str):
    intent_cols = [f"intent_{k}" for k in INTENTS]
    export_cols = (
        ["sentence_id", "review_id", "text", "primary_intent", "intent_count"]
        + intent_cols
        + ["escalation_risk", "human_score", "human_label",
           "vader_neg", "vader_neu", "vader_pos", "vader_compound", "vader_label"]
    )
    df[export_cols].to_csv(path, index=False)
    print(f"\n   Results CSV saved → {path}")


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 72)
    print("  VADER Sentiment Analysis — Customer Support Dataset")
    print("=" * 72 + "\n")

    df = load_amazon_reviews()
    df = tag_intents(df)
    df = run_vader(df)

    results = evaluate(df)

    build_dashboard(df, results, "vader_cs_dashboard.png")
    print_report(results)
    save_csv(df, "vader_cs_results.csv")

    print("\n✅  Complete!")
    print("   vader_cs_dashboard.png  — 8-panel analytics dashboard")
    print("   vader_cs_results.csv    — full scored dataset with intents\n")


if __name__ == "__main__":
    main()