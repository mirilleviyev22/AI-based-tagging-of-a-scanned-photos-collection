# compare_description_vs_semantic_summary.py
# Compares model "description" to manual "semantic summery" per image.

import os
import math
import re
import pandas as pd


MODEL_FILE = r"image_results_20250814_140919.xlsx"  # has columns: image_path, description
GT_FILE    = r"tagging pictures2.xlsx"              # has first column = filename + 'semantic summery'
OUT_FILE   = r"text_eval_description_vs_semantic_summary.xlsx"
SBERT_MODEL = "all-MiniLM-L6-v2"   # fast & good
USE_BERTSCORE = True
# --------------------------

def normalize_stem(x: str) -> str:
    stem = os.path.splitext(os.path.basename(str(x)))[0]
    return stem.strip().lower()

def clean_text(x: str) -> str:
    return re.sub(r"\s+", " ", str(x).strip())

def safe_import_sentence_transformers():
    try:
        from sentence_transformers import SentenceTransformer, util
        return SentenceTransformer, util
    except Exception as e:
        print(f"[WARN] sentence_transformers not available: {e}")
        return None, None

def sbert_cosine_batch(a_texts, b_texts, model_name=SBERT_MODEL):
    SentenceTransformer, util = safe_import_sentence_transformers()
    if SentenceTransformer is None:
        return [math.nan] * len(a_texts)
    model = SentenceTransformer(model_name)
    a_emb = model.encode(a_texts, convert_to_tensor=True, show_progress_bar=False)
    b_emb = model.encode(b_texts, convert_to_tensor=True, show_progress_bar=False)
    return util.cos_sim(a_emb, b_emb).diagonal().cpu().tolist()

def safe_import_bertscore():
    if not USE_BERTSCORE:
        return None
    try:
        from bert_score import score as bert_score
        return bert_score
    except Exception as e:
        print(f"[WARN] bert_score not available: {e}")
        return None

def bertscore_batch(a_texts, b_texts, lang="en"):
    bs = safe_import_bertscore()
    if bs is None:
        n = len(a_texts)
        return [math.nan]*n, [math.nan]*n, [math.nan]*n
    P, R, F1 = bs(a_texts, b_texts, lang=lang, verbose=False)
    return [p.item() for p in P], [r.item() for r in R], [f.item() for f in F1]

# ---- Load model outputs ----
model = pd.read_excel(MODEL_FILE)
if not {"image_path", "description"}.issubset(model.columns):
    raise ValueError("MODEL_FILE must contain columns: image_path, description")

model = model.copy()
model["stem"] = model["image_path"].map(normalize_stem)
model["description_txt"] = model["description"].map(clean_text)

# ---- Load manual GT ----
gt = pd.read_excel(GT_FILE)
first_col = gt.columns[0]  # filename column
# try both spellings just in case
sem_col = "semantic summery" if "semantic summery" in gt.columns else \
          ("semantic summary" if "semantic summary" in gt.columns else None)
if sem_col is None:
    raise ValueError("GT file must contain a column named 'semantic summery' (or 'semantic summary').")

gt = gt.copy()
gt["stem"] = gt[first_col].map(normalize_stem)
gt["semantic_txt"] = gt[sem_col].map(lambda s: clean_text("" if pd.isna(s) else s))

# ---- Join on filename stem ----
merged = model.merge(gt, on="stem", how="inner", suffixes=("_model", "_gt"))
if merged.empty:
    raise RuntimeError("No matching filenames between model and GT after normalization.")

# ---- Similarity metrics: description (model) vs semantic summery (GT) ----
a_texts = merged["description_txt"].tolist()
b_texts = merged["semantic_txt"].tolist()

merged["sbert_cosine"] = sbert_cosine_batch(a_texts, b_texts, model_name=SBERT_MODEL)
P, R, F1 = bertscore_batch(merged["description"].tolist(), merged[sem_col].tolist(), lang="en")
merged["bert_P"] = P
merged["bert_R"] = R
merged["bert_F1"] = F1

# ---- Summary ----
def mean_ignore_nan(s):
    s = pd.to_numeric(s, errors="coerce")
    return float(s.mean()) if s.notna().any() else math.nan

summary = pd.DataFrame([
    {"metric": "SBERT cosine (mean)", "value": mean_ignore_nan(merged["sbert_cosine"])},
    {"metric": "BERTScore P (mean)",  "value": mean_ignore_nan(merged["bert_P"])},
    {"metric": "BERTScore R (mean)",  "value": mean_ignore_nan(merged["bert_R"])},
    {"metric": "BERTScore F1 (mean)", "value": mean_ignore_nan(merged["bert_F1"])},
    {"metric": "Pairs matched (rows)", "value": len(merged)}
])

# ---- Save Excel ----
cols = [
    "stem", "image_path", "description", sem_col,
    "sbert_cosine", "bert_P", "bert_R", "bert_F1"
]
with pd.ExcelWriter(OUT_FILE, engine="xlsxwriter") as xl:
    merged[cols].to_excel(xl, index=False, sheet_name="results")
    summary.to_excel(xl, index=False, sheet_name="summary")

print(f"✅ Wrote {OUT_FILE}")
print(summary.to_string(index=False))
