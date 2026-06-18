"""
Run this from your project root:
    python test_predict_v2.py

Tests both exact medical terms AND casual everyday language.
"""
import sys, os
# Adjust this path to your project root
PROJECT_ROOT = r"C:\Users\Kamran Computers\Downloads\first_aid_ai"
sys.path.insert(0, PROJECT_ROOT)

from src.predict import predict

tests = [
    # casual phrasing                     expected keyword in Situation
    ("my hand got burnt on stove",         "Burn"),
    ("kid has high fever",                 "Fever"),
    ("baby is choking",                    "Choking"),
    ("dog bit me",                         "Animal Bite"),
    ("snake bite on leg",                  "Snake Bite"),
    ("can't breathe",                      "Breathing"),
    ("someone collapsed not breathing",    "Cardiac Arrest"),
    ("chest pain and sweating",            "Heart"),
    ("face drooping arm weak",             "Stroke"),
    ("feel dizzy shaky and hungry diabetic","Hypoglycemia"),
    ("nose is bleeding",                   "Nosebleed"),
    ("swallowed bleach",                   "Poisoning"),
    ("electric shock from socket",         "Electric"),
    ("severe headache one side nausea",    "Migraine"),
    ("bee sting swollen",                  "Insect"),
    ("chemical in my eye",                 "Eye"),
    ("person fainted and fell",            "Fainting"),
    ("broken arm after fall",              "Fracture"),
    ("stomach hurts and throwing up",      "Vomiting"),
    ("ear pain something stuck",           "Ear"),
    ("panic attack racing heart",          "Panic"),
    ("sunburn peeling blisters",           "Sunburn"),
    ("ankle sprain from running",          "Sprain"),
    ("asthma inhaler not working",         "Asthma"),
    ("toothache swollen jaw",              "Dental"),
]

print(f"{'QUERY':<45} {'MATCHED':<38} {'SCORE':>7}")
print("─" * 95)

correct = 0
for query, expected_keyword in tests:
    result, score = predict(query)
    if "Error" in result:
        matched = "[NO MATCH]"
        hit = False
    else:
        matched = result.get("Situation", "?")
        hit = expected_keyword.lower() in matched.lower()
        if hit:
            correct += 1

    flag = "✅" if hit else "❌"
    bar  = "🟢" if score >= 0.60 else ("🟡" if score >= 0.35 else "🔴")
    print(f"{flag} {query:<43} → {matched:<38} {bar} {round(score*100,1):>5}%")

total = len(tests)
pct   = round(correct / total * 100)
print(f"\n{'─'*95}")
print(f"Accuracy: {correct}/{total} = {pct}%")
if pct >= 90:
    print("🟢 Excellent matching quality")
elif pct >= 75:
    print("🟡 Good — a few edge cases to review")
else:
    print("🔴 Needs further tuning — check ❌ rows above")
