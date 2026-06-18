"""
train.py — Hybrid Matching Trainer
Builds TWO complementary indexes:
  1. Semantic embeddings  (all-MiniLM-L6-v2)  — catches meaning/paraphrasing
  2. Keyword/TF-IDF index (sklearn)            — catches exact symptom words

Both are saved to /models/ and used together at prediction time.
"""
import os
import re
import pickle
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "first_aid.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")


def build_master_text(row: pd.Series) -> str:
    """
    Combine Situation + Symptoms into one rich text chunk.
    The situation name is repeated several times so it gets higher weight
    in both the TF-IDF and the semantic vector.
    """
    situation = str(row.get("Situation", "")).strip()
    symptoms  = str(row.get("Symptoms",  "")).strip()
    # Repeat the situation name to boost its weight
    return f"{situation} {situation} {situation} {symptoms}"


def train_model():
    os.makedirs(MODELS_DIR, exist_ok=True)

    if not os.path.exists(DATA_PATH):
        print(f"❌  Dataset missing at {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH, on_bad_lines="skip")
    df.columns = df.columns.str.strip()
    num_cases = len(df)
    print(f"📂  Loaded {num_cases} first-aid cases.")

    # ── Build master text for each row ─────────────────────────────────────
    df["Master_Text"] = df.apply(build_master_text, axis=1)

    # ── 1. Semantic model ───────────────────────────────────────────────────
    print("🧠  Loading semantic model (all-MiniLM-L6-v2)…")
    sem_model = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"🧬  Encoding {num_cases} cases into semantic vectors…")
    X_semantic = sem_model.encode(
        df["Master_Text"].tolist(),
        show_progress_bar=True,
        normalize_embeddings=True,   # unit-normalised → dot-product == cosine
    )

    # ── 2. TF-IDF keyword model ─────────────────────────────────────────────
    print("📊  Building TF-IDF keyword index…")
    tfidf = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),   # unigrams + bigrams
        min_df=1,
        sublinear_tf=True,
    )
    X_tfidf = tfidf.fit_transform(df["Master_Text"])

    # ── Save everything ─────────────────────────────────────────────────────
    artifacts = {
        "semantic_model.pkl": sem_model,
        "X_embeddings.pkl":   X_semantic,
        "tfidf_model.pkl":    tfidf,
        "X_tfidf.pkl":        X_tfidf,
        "data.pkl":           df,
    }
    for filename, obj in artifacts.items():
        with open(os.path.join(MODELS_DIR, filename), "wb") as f:
            pickle.dump(obj, f)

    print(f"✅  Training complete — {num_cases} situations indexed (semantic + TF-IDF)!")


if __name__ == "__main__":
    train_model()
