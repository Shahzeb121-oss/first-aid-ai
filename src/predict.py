"""
predict.py — Hybrid Matcher with Query Expansion
─────────────────────────────────────────────────
Three-layer matching strategy:

  1. Query Expansion  — maps everyday words to medical vocabulary
                        e.g. "baby" → "infant", "can't breathe" → "shortness of breath"

  2. Semantic score   — all-MiniLM-L6-v2 catches meaning & paraphrasing

  3. TF-IDF score     — catches exact symptom keywords

Final: blended score = 0.55 × semantic + 0.45 × tfidf
"""
import os
import re
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR  = os.path.join(BASE_DIR, "models")

SEMANTIC_WEIGHT = 0.55
KEYWORD_WEIGHT  = 0.45
MIN_CONFIDENCE  = 0.15

# ── Query Expansion Dictionary ─────────────────────────────────────────────
# Maps common user words → medical terms that appear in the CSV.
# Keys are plain words/phrases users actually type.
# Values are additional words appended to the query before matching.
EXPANSIONS = {
    # People
    "baby":          "infant child",
    "kid":           "child infant children",
    "toddler":       "infant child",
    "elderly":       "adult",
    "pregnant":      "pregnancy",
    # Burns
    "burnt":         "burn burned burning",
    "scald":         "burn hot water steam",
    "fire":          "burn flame smoke inhalation",
    # Breathing
    "can't breathe": "difficulty breathing shortness of breath respiratory",
    "cant breathe":  "difficulty breathing shortness of breath respiratory",
    "short of breath": "difficulty breathing shortness of breath",
    "wheezing":      "asthma breathing difficulty",
    "breathless":    "shortness of breath difficulty breathing",
    # Bleeding
    "blood":         "bleeding wound cut hemorrhage",
    "bleeding a lot": "heavy bleeding uncontrolled hemorrhage severe cut",
    "bleeding nose":  "nosebleed epistaxis nasal bleeding",
    # Chest
    "chest hurts":   "chest pain tightness pressure heart",
    "heart pain":    "chest pain heart attack cardiac",
    "heart attack":  "chest pain crushing pressure radiating arm jaw",
    # Head
    "head hurts":    "headache migraine tension",
    "bad headache":  "severe headache migraine thunderclap",
    # Stomach / Abdomen
    "stomach ache":  "abdominal pain nausea vomiting",
    "stomach hurts": "abdominal pain nausea cramping",
    "threw up":      "vomiting nausea",
    "throwing up":   "vomiting nausea",
    "puking":        "vomiting nausea",
    "puke":          "vomiting nausea",
    "diarrhoea":     "diarrhea loose stools watery",
    "loose motions": "diarrhea loose stools watery",
    # Consciousness
    "passed out":    "fainted unconscious syncope collapse",
    "fainted":       "syncope fainting unconscious",
    "knocked out":   "unconscious loss of consciousness head injury",
    "unresponsive":  "unconscious no breathing cardiac arrest",
    "not responding": "unconscious unresponsive",
    # Allergic
    "allergic reaction": "allergy anaphylaxis hives swelling",
    "swollen throat":    "anaphylaxis throat swelling difficulty breathing",
    "epipen":            "anaphylaxis epinephrine severe allergy",
    # Bites / Stings
    "dog bite":      "animal bite wound infection rabies",
    "cat bite":      "animal bite wound infection",
    "bee sting":     "insect sting bite venom",
    "wasp":          "insect sting bite venom",
    "spider":        "insect bite venom",
    "snake":         "snake bite venom antivenom",
    "jellyfish":     "jellyfish sting marine venom",
    # Eyes
    "eye hurts":     "eye pain injury irritation",
    "something in eye": "foreign object eye cornea",
    "chemical in eye":  "chemical splash eye burn irrigation",
    # Poisoning
    "swallowed":     "ingested poisoning overdose",
    "ate something": "ingested poisoning",
    "overdose":      "drug overdose medication poisoning",
    "bleach":        "chemical poisoning corrosive ingested",
    "gas leak":      "carbon monoxide inhaled poisoning fumes",
    "smoke":         "smoke inhalation carbon monoxide fire",
    # Fractures / Injuries
    "broken bone":   "fracture break bone deformity",
    "sprained":      "sprain strain ligament joint",
    "dislocated":    "dislocation joint deformity",
    "back injury":   "spinal back fracture injury",
    "neck injury":   "spinal cervical injury",
    # Seizures
    "fitting":       "seizure convulsions epilepsy",
    "shaking":       "seizure convulsions tremor",
    "convulsing":    "seizure convulsions tonic clonic",
    # Diabetes
    "diabetic":      "diabetes blood sugar glucose",
    "low sugar":     "hypoglycemia low blood glucose diabetic",
    "high sugar":    "hyperglycemia high blood glucose diabetic",
    # Heart
    "stroke":        "face drooping arm weakness speech slurred FAST",
    "heart stopped": "cardiac arrest CPR no pulse",
    # Temperature
    "high temperature": "fever elevated temperature",
    "temperature":      "fever body temperature",
    "hot to touch":     "fever elevated temperature",
    "cold":             "hypothermia frostbite cold temperature",
    "frozen":           "frostbite hypothermia cold",
    # Teeth
    "toothache":     "dental pain tooth infection",
    "tooth fell out": "avulsed knocked out tooth dental",
    # Ear / Nose
    "ear pain":      "ear injury foreign object",
    "nose broken":   "nasal fracture nose injury",
    # Electric
    "electric shock": "electrical shock burn entry exit",
    "electrocuted":   "electrical burn cardiac arrhythmia",
    # Miscellaneous
    "uti":           "urinary tract infection burning urination",
    "panic":         "panic attack anxiety chest tightness",
    "anxiety attack": "panic attack anxiety",
    "frostbite":     "frozen skin extremities cold exposure",
    "sunburn":       "sun burn red skin UV",
}


def expand_query(query: str) -> str:
    """
    Appends medical synonym terms to the user query so both
    semantic and TF-IDF models have richer vocabulary to work with.
    """
    q_lower = query.lower()
    extra_terms = []
    for trigger, expansion in EXPANSIONS.items():
        if trigger in q_lower:
            extra_terms.append(expansion)
    if extra_terms:
        return query + " " + " ".join(extra_terms)
    return query


def _load_models():
    def _load(name):
        path = os.path.join(MODELS_DIR, name)
        if not os.path.exists(path):
            raise FileNotFoundError(name)
        with open(path, "rb") as f:
            return pickle.load(f)

    return (
        _load("semantic_model.pkl"),
        _load("X_embeddings.pkl"),
        _load("tfidf_model.pkl"),
        _load("X_tfidf.pkl"),
        _load("data.pkl"),
    )


def predict(query: str):
    """
    Returns (result_dict, blended_score).
    result_dict has an "Error" key on failure.
    """
    try:
        sem_model, X_sem, tfidf, X_tfidf, df = _load_models()
    except FileNotFoundError as e:
        return {"Error": f"Model file missing ({e}). Please run train.py first."}, 0.0

    raw = str(query).strip()
    if not raw:
        return {"Error": "Empty query."}, 0.0

    # ── Query expansion ────────────────────────────────────────────────────
    expanded = expand_query(raw)

    # ── Semantic score ─────────────────────────────────────────────────────
    q_sem    = sem_model.encode([expanded], normalize_embeddings=True)
    sem_scores = (q_sem @ X_sem.T)[0]          # shape (n,)

    # ── TF-IDF keyword score ───────────────────────────────────────────────
    q_tfidf      = tfidf.transform([expanded])
    tfidf_scores = cosine_similarity(q_tfidf, X_tfidf)[0]   # shape (n,)

    # ── Blend ──────────────────────────────────────────────────────────────
    blended  = SEMANTIC_WEIGHT * sem_scores + KEYWORD_WEIGHT * tfidf_scores
    best_idx  = int(np.argmax(blended))
    best_score = float(blended[best_idx])

    # ── Confidence gate ────────────────────────────────────────────────────
    if best_score < MIN_CONFIDENCE:
        return {
            "Error": (
                f"No closely matching situation found "
                f"(confidence {round(best_score * 100, 1)}%). "
                "Please describe symptoms in more detail, "
                "or call emergency services if this is urgent."
            )
        }, best_score

    result = df.iloc[best_idx].to_dict()
    return result, best_score

