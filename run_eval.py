import os
import json
import pandas as pd
from tqdm import tqdm
from Evaluation.metric import TextMetrics
import google.generativeai as genai
from dotenv import load_dotenv

# -------------------------------
#  CONFIG
# -------------------------------

load_dotenv(".env")

IDEAL_PATH = "eval_assets/ideal_answers.json"
RESULTS_PATH = "eval_assets/results.json"

MODEL_TYPES = ["default", "rag", "index", "hybrid"]

# Load Gemini API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

judge_model = genai.GenerativeModel("gemini-2.5-flash")

# -------------------------------
#  LLM JUDGE PROMPT
# -------------------------------
def llm_judge(ideal, pred):
    """Ask Gemini to evaluate an answer and return structured scores."""

    prompt = f"""
You are an evaluation judge. Score the answer STRICTLY in JSON.

Return ONLY valid JSON with NO commentary, NO markdown, NO backticks.

Format:
{{
  "creativity": number,
  "coherence": number,
  "factuality": number,
  "completeness": number,
  "clarity": number,
  "final_score": number
}}

IDEAL ANSWER:
{ideal}

MODEL ANSWER:
{pred}
"""
    try:
        response = judge_model.generate_content(prompt)
        text = response.text

        # Safely parse JSON from the LLM
        scores = json.loads(text)
        return scores

    except Exception as e:
        print("Judge Error:", e)
        # return a blank score dict
        return {
            "creativity": 0,
            "coherence": 0,
            "factuality": 0,
            "completeness": 0,
            "clarity": 0,
            "final_score": 0
        }


# -------------------------------
# Load ideal & model answers
# -------------------------------
with open(IDEAL_PATH, "r") as f:
    ideal_answers = json.load(f)

with open(RESULTS_PATH, "r") as f:
    model_answers = json.load(f)

questions = list(ideal_answers.keys())

metrics = TextMetrics()
final_rows = []

print("\nEvaluating all answers with prefixed column names...")

# -------------------------------
# Main loop
# -------------------------------
for q in tqdm(questions, desc="Evaluating"):
    ideal = ideal_answers[q]
    model_pack = model_answers.get(q, {})

    row = {"question": q}

    for mtype in MODEL_TYPES:

        pack = model_pack.get(mtype, {})
        pred = pack.get("answer", "")
        latency = pack.get("latency", None)

        # ---- Classical Metrics ----
        met = metrics.compute_metrics(ideal, pred)

        row[f"{mtype}_bleu"] = met["BLEU"]
        row[f"{mtype}_rouge1"] = met["ROUGE-1"]
        row[f"{mtype}_rougel"] = met["ROUGE-L"]
        row[f"{mtype}_cosine"] = met["CosineSim"]
        row[f"{mtype}_euclidean"] = met["EuclideanDist"]
        row[f"{mtype}_tfidf"] = met["TFIDF_Cosine"]
        row[f"{mtype}_latency"] = latency
        row[f"{mtype}_words"] = len(pred.split(" "))
        row[f"{mtype}_chars"] = len(pred)

        # ---- LLM-as-Judge Metrics ----
        judge_scores = llm_judge(ideal, pred)

        row[f"{mtype}_judge_creativity"]   = judge_scores["creativity"]
        row[f"{mtype}_judge_coherence"]    = judge_scores["coherence"]
        row[f"{mtype}_judge_factuality"]   = judge_scores["factuality"]
        row[f"{mtype}_judge_completeness"] = judge_scores["completeness"]
        row[f"{mtype}_judge_clarity"]      = judge_scores["clarity"]
        row[f"{mtype}_judge_final"]        = judge_scores["final_score"]

    final_rows.append(row)

# -------------------------------
# Save dataframe
# -------------------------------
df = pd.DataFrame(final_rows)

df.to_csv("eval_results/evaluation_with_llm_judge.csv", index=False)
df.to_json("eval_results/evaluation_with_llm_judge.json", indent=4)

print("\n✔ Saved evaluation_with_llm_judge.csv and evaluation_with_llm_judge.json")
print(df.columns)
