<div align="center">

# 🚑 AI First Aid Emergency Assistant

**Instant, symptom-aware first aid guidance — powered by hybrid semantic + keyword AI**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![SentenceTransformers](https://img.shields.io/badge/SentenceTransformers-all--MiniLM--L6--v2-brightgreen)](https://www.sbert.net/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)



</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [How It Works](#-how-it-works)
- [Features](#-features)
- [Covered Emergencies](#-covered-emergencies-67-situations)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [Architecture](#-architecture)
- [Query Expansion](#-query-expansion)
- [Configuration](#-configuration)
- [Running Tests](#-running-tests)
- [Requirements](#-requirements)
- [Disclaimer](#-disclaimer)

---

## 🔍 Overview

**AI First Aid Emergency Assistant** is a locally-running Streamlit web app that takes a plain-English description of a medical emergency — typed the way a real person would type it in a panic — and instantly returns:

- ✅ Identified medical situation
- ⚡ Severity level (Minor → Life-Threatening)
- 📋 Step-by-step first aid instructions
- 💊 Recommended medicines and supplies
- 🚨 Emergency triggers (when to call 1122/115)
- ⚠️ Critical medical warnings

It runs **100% offline** after the one-time model download. No API keys. No internet required at query time.

---

## 🧠 How It Works

The system uses a **three-layer hybrid matching pipeline** — not a simple keyword search, and not a black-box LLM:

```
User query  ──►  Query Expansion  ──►  Semantic Score  ──┐
                 (synonym mapping)     (all-MiniLM-L6)   ├──► Blended Score ──► Best Match
                                   ──►  TF-IDF Score   ──┘
                                       (keyword overlap)
```

1. **Query Expansion** — maps everyday words to medical vocabulary before matching.
   `"baby choking"` becomes `"baby choking infant child"` so the model finds *Choking (Infant)* correctly.

2. **Semantic Similarity** — `all-MiniLM-L6-v2` encodes meaning. Catches paraphrasing and conceptual similarity even when exact words differ.

3. **TF-IDF Keyword Matching** — catches exact medical terms and symptom words. Anchors the result when the user types a specific term like `"anaphylaxis"` or `"snake bite"`.

4. **Blended Score** — both signals are combined:
   ```
   final_score = 0.55 × semantic_score + 0.45 × tfidf_score
   ```
   Each layer compensates for the other's blind spots.

---

## ✨ Features

| Feature | Detail |
|---|---|
| 🗣️ Natural language input | Type the way you'd speak: *"my kid burned their hand"* |
| 🔁 Query expansion | 60+ synonym mappings bridge casual language to medical terms |
| 🧬 Semantic AI | Sentence-transformer model understands meaning, not just keywords |
| 📊 Keyword matching | TF-IDF ensures exact symptom terms are never missed |
| 🎚️ Confidence scoring | Colour-coded: 🟢 High / 🟡 Moderate / 🔴 Low |
| 🚦 Severity classification | 7 levels from Minor to Life-Threatening, all correctly labelled |
| 💊 Supplies & medicines | Specific OTC and emergency medicines per situation |
| 🚨 Emergency triggers | Clear conditions for when to call emergency services |
| ⚠️ Medical warnings | Safety-critical *do NOTs* for each situation |
| 🖥️ Fully offline | No API calls at inference time — runs on your machine |
| ⚡ Fast | Typical query response in a few seconds after model load

---

## 🏥 Covered Emergencies — 67 Situations

<details>
<summary>Click to expand full list</summary>

**Burns & Wounds**
- Minor Burn (1st Degree) · Severe Burn (2nd & 3rd Degree) · Chemical Burn · Electrical Burn
- Cut / Wound (Minor) · Deep Laceration / Severe Cut · Bleeding from Ear

**Bites & Stings**
- Insect Bite / Sting · Snake Bite · Animal Bite · Human Bite · Jellyfish Sting

**Cardiac & Circulatory**
- Cardiac Arrest · Heart Attack · Shock · Chest Pain (Unknown Cause)

**Neurological**
- Stroke · Seizure / Convulsions · Concussion / Head Injury · Fainting / Syncope · Migraine · Headache (Tension) · Panic Attack

**Respiratory**
- Choking (Adult/Child) · Choking (Infant) · Asthma Attack · Difficulty Breathing · Carbon Monoxide Poisoning · Breathing in Smoke (Fire)

**Diabetes**
- Diabetic Hypoglycemia · Diabetic Hyperglycemia · Diabetic Foot Wound

**Poisoning & Overdose**
- Poisoning (Ingested) · Poisoning (Inhaled) · Poisoning (Skin Contact) · Drug Overdose

**Environmental**
- Heat Stroke · Heat Exhaustion · Hypothermia · Hypothermia (Mild) · Frostbite · Drowning / Near-Drowning · Sunburn

**Allergic**
- Allergy (Mild) · Anaphylaxis (Severe Allergic Reaction)

**Musculoskeletal**
- Fracture / Broken Bone · Sprain / Strain · Dislocation · Back / Spinal Injury · Nose Injury

**Eyes, Ears & Dental**
- Eye Injury (Chemical) · Eye Injury (Foreign Object) · Eye Allergy / Irritation
- Ear Injury / Foreign Object · Dental Pain / Toothache · Knocked-Out Tooth (Avulsed)

**Gastrointestinal**
- Diarrhea · Vomiting · Nosebleed · Epistaxis (Anterior)

**Other**
- Fever (Adults) · Fever (Children) · Electric Shock (Minor) · UTI
- Pregnancy Emergency (Miscarriage) · Pregnancy Emergency (Pre-eclampsia)
- Hypothyroidism Crisis (Myxedema Coma)

</details>

---

## 📁 Project Structure

```
first_aid_ai/
│
│
├── data/
│   └── first_aid.csv              # Dataset — 67 first-aid situations
│
├── models/                        # Auto-generated by train.py (git-ignored)
│   ├── semantic_model.pkl         # SentenceTransformer model
│   ├── X_embeddings.pkl           # Semantic vectors (67 × 384)
│   ├── tfidf_model.pkl            # TF-IDF vectorizer
│   ├── X_tfidf.pkl                # TF-IDF document matrix
│   └── data.pkl                  # Full DataFrame
│
├── src/
│   ├── train.py                  # Training pipeline — builds all 5 model artifacts
│   ├── predict.py                # Hybrid inference engine (expansion + semantic + tfidf)
│   └── utils.py                  # Severity icon helper
│
├── app.py                        # Streamlit UI entry point
├── test_predict.py               # 25-case accuracy test suite
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/your-username/first-aid-ai.git
cd first-aid-ai
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

PyTorch must be installed separately as the CPU-only build to avoid DLL crashes on Windows:

```bash
# Install CPU-only PyTorch first
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu

# Then install everything else
pip install -r requirements.txt
```

### 4. Train the model

```bash
python src/train.py
```

Expected output:
```
📂  Loaded 67 first-aid cases.
🧠  Loading semantic model (all-MiniLM-L6-v2)…
🧬  Encoding 67 cases into semantic vectors…
📊  Building TF-IDF keyword index…
✅  Training complete — 67 situations indexed (semantic + TF-IDF)!
```

> First run downloads `all-MiniLM-L6-v2` (~90 MB) from HuggingFace. Subsequent runs use the cached version.

### 5. Launch the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 💻 Usage

Type your emergency in plain language — exactly how you would describe it to someone:

| You type | Matched situation |
|---|---|
| `my hand got burnt on a hot pan` | Minor Burn (1st Degree) |
| `kid has high fever` | Fever (Children) |
| `baby is choking` | Choking (Infant under 1 year) |
| `can't breathe, chest tight` | Asthma Attack |
| `someone collapsed not breathing` | Cardiac Arrest |
| `face drooping arm weak` | Stroke |
| `swallowed bleach` | Poisoning (Ingested) |
| `dog bit my arm` | Animal Bite |
| `feel shaky and dizzy, diabetic` | Diabetic Hypoglycemia |

---

## 🏗️ Architecture

### Training (`src/train.py`)

```
first_aid.csv
     │
     ▼
 Clean & parse columns
     │
     ▼
 Build Master_Text = Situation × 3 + Symptoms
     │
     ├──► SentenceTransformer.encode()  ──► X_embeddings.pkl  (semantic)
     │
     └──► TfidfVectorizer.fit_transform() ──► X_tfidf.pkl     (keyword)
```

The situation name is **repeated 3 times** in `Master_Text` to give it higher weight in both indexes — so `"snake bite"` strongly anchors to the Snake Bite row.

### Inference (`src/predict.py`)

```
Raw query
   │
   ▼
expand_query()          # e.g. "baby" → "baby infant child"
   │
   ├──► sem_model.encode(expanded)  → cosine vs X_embeddings  → sem_scores
   │
   └──► tfidf.transform(expanded)  → cosine vs X_tfidf        → tfidf_scores
                                                                      │
                                           blended = 0.55×sem + 0.45×tfidf
                                                                      │
                                                              argmax → best row
```

### Confidence gate

Queries scoring below `MIN_CONFIDENCE = 0.15` return an explicit "no match" error instead of a misleading low-confidence result.

---

## 🔤 Query Expansion

`predict.py` contains a hand-curated dictionary of **60+ trigger phrases** mapping everyday language to the medical vocabulary used in the dataset:

```python
EXPANSIONS = {
    "baby":           "infant child",
    "can't breathe":  "difficulty breathing shortness of breath respiratory",
    "threw up":       "vomiting nausea",
    "swallowed":      "ingested poisoning overdose",
    "bleach":         "chemical poisoning corrosive ingested",
    "electric shock": "electrical shock burn entry exit",
    "passed out":     "fainted unconscious syncope collapse",
    # ... 50+ more
}
```

To add support for a new phrase, simply add a key-value pair to `EXPANSIONS` in `src/predict.py`. No retraining required.

---

## ⚙️ Configuration

All tunable parameters are at the top of `src/predict.py`:

```python
SEMANTIC_WEIGHT = 0.55   # Weight given to semantic similarity (0.0 – 1.0)
KEYWORD_WEIGHT  = 0.45   # Weight given to TF-IDF keyword match (must sum to 1.0)
MIN_CONFIDENCE  = 0.15   # Minimum blended score to return a result
```

Tuning tips:
- Increase `SEMANTIC_WEIGHT` if users tend to paraphrase or describe symptoms abstractly
- Increase `KEYWORD_WEIGHT` if users tend to type specific medical terms
- Lower `MIN_CONFIDENCE` to be more permissive; raise it to reduce false positives

---

## 🧪 Running Tests

A 25-case accuracy suite is included:

```bash
python test_predict.py
```

Sample output:
```
QUERY                                         MATCHED                              SCORE
───────────────────────────────────────────────────────────────────────────────────────
✅ my hand got burnt on stove               → Minor Burn (1st Degree)             🟢 74.2%
✅ baby is choking                          → Choking (Infant under 1 year)       🟢 68.5%
✅ can't breathe                            → Difficulty Breathing (General)      🟡 51.3%
✅ swallowed bleach                         → Poisoning (Ingested)                🟡 43.8%
✅ face drooping arm weak                   → Stroke                              🟢 79.1%
...
───────────────────────────────────────────────────────────────────────────────────────
Accuracy: 23/25 = 92%
🟢 Excellent matching quality
```

Any `❌` failures indicate a missing expansion entry. Add the phrase to `EXPANSIONS` in `src/predict.py` — no retraining needed.

---

## 📦 Requirements

```
# ── Web UI ────────────────────────────────────────────────────────────────────
streamlit==1.35.0

# ── ML / NLP ──────────────────────────────────────────────────────────────────
# NOTE: PyTorch is NOT listed here — install it separately before this file:
#   pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu
#
# The default `pip install torch` pulls the CUDA/GPU build which crashes on
# most Windows machines with a DLL initialization error.
sentence-transformers==2.7.0
scikit-learn==1.5.0

# ── Data ──────────────────────────────────────────────────────────────────────
pandas==2.2.2
numpy==1.26.4

# ── Serialization / Compatibility ─────────────────────────────────────────────
protobuf==4.25.3

```

> **PyTorch is not in `requirements.txt` intentionally.** The default `pip install torch` pulls the CUDA/GPU build which requires specific Microsoft Visual C++ DLLs and will crash on most machines with `DLL initialization failed`. Install the CPU-only build manually first:

```bash
# Step 1 — Install CPU-only PyTorch (works on all machines, no GPU/CUDA needed)
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu

# Step 2 — Install all other dependencies
pip install -r requirements.txt
```

> **Python 3.10.4** is required.

---

## ⚠️ Disclaimer

> This application is intended as a **first aid reference tool** to assist during emergencies while waiting for professional help. It is **not a substitute for professional medical advice, diagnosis, or treatment.**
>
> - Always call emergency services (1122 or 115) in life-threatening situations
> - The AI matches based on described symptoms — it cannot examine a patient
> - Confidence scores are similarity measures, not medical certainty
> - Do not rely solely on this tool for critical medical decisions

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">


**If this helped you — give it a ⭐**

</div> 
