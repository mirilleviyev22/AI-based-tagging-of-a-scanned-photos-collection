import os
import pandas as pd

GT = "image_pairs_test1.xlsx"                       # manual GT
PAIRS = "similarities_20251106_190251.xlsx"         # model pairs
THR = 0.85
OUT = f"verification_report_thr_{str(THR).replace('.','_')}.xlsx"

def normalize_pairs(df, c1, c2):
    a = df[c1].astype(str).map(lambda p: os.path.basename(p).lower())
    b = df[c2].astype(str).map(lambda p: os.path.basename(p).lower())
    return pd.DataFrame([tuple(sorted(x)) for x in zip(a, b)], columns=["a","b"])

# --- Load GT ---
gt = pd.read_excel(GT)
if "true_label" not in gt.columns:
    gt["true_label"] = (gt["true_count"].astype(float) > 0).astype(int)

gt[["a","b"]] = normalize_pairs(gt, "img1", "img2")
gt = gt[["a","b","true_label"]].drop_duplicates()
n_gt = len(gt)

# --- Load model pairs ---
if PAIRS.lower().endswith((".xlsx", ".xls")):
    pairs = pd.read_excel(PAIRS)
elif PAIRS.lower().endswith(".parquet"):
    pairs = pd.read_parquet(PAIRS)
else:
    pairs = pd.read_csv(PAIRS)

pairs[["a","b"]] = normalize_pairs(pairs, "image1_path", "image2_path")
pairs = pairs[["a","b","similarity_score"]].dropna(subset=["a","b","similarity_score"]).drop_duplicates()

# --- Merge & predict ---
m = gt.merge(pairs, on=["a","b"], how="left")
m["sim"]  = m["similarity_score"]
m["pred"] = (m["sim"] >= THR).astype(int).fillna(0)   # missing sim → predict 0
m["true"] = m["true_label"].astype(int)

tp = int(((m["pred"] == 1) & (m["true"] == 1)).sum())
tn = int(((m["pred"] == 0) & (m["true"] == 0)).sum())
fp = int(((m["pred"] == 1) & (m["true"] == 0)).sum())
fn = int(((m["pred"] == 0) & (m["true"] == 1)).sum())

total = tp + tn + fp + fn
acc = (tp + tn) / total if total else 0.0
prec = tp / (tp + fp) if (tp + fp) else 0.0
rec  = tp / (tp + fn) if (tp + fn) else 0.0
f1   = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0

# Coverage: GT pairs that had a similarity score computed
covered = m["sim"].notna().sum()
coverage_rate = covered / n_gt if n_gt else 0.0

print(f"GT pairs: {n_gt} | Covered by model: {covered} ({coverage_rate:.1%})")
print(f"Threshold = {THR}")
print(f"TP={tp}  FP={fp}  FN={fn}  TN={tn}")
print(f"Accuracy={acc:.6f}  Precision={prec:.6f}  Recall={rec:.6f}  F1={f1:.6f}")

# Save a small Excel report (metrics + FP/FN examples)
summary = pd.DataFrame([{
    "threshold": THR,
    "GT_pairs": n_gt,
    "covered_pairs": covered,
    "coverage_rate": coverage_rate,
    "TP": tp, "FP": fp, "FN": fn, "TN": tn,
    "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1
}])

fp_rows = m[(m["pred"] == 1) & (m["true"] == 0)][["a","b","sim"]].copy()
fn_rows = m[(m["pred"] == 0) & (m["true"] == 1)][["a","b","sim"]].copy()

with pd.ExcelWriter(OUT, engine="xlsxwriter") as xl:
    summary.to_excel(xl, index=False, sheet_name="summary")
    m[["a","b","sim","true","pred"]].to_excel(xl, index=False, sheet_name="comparison")
    if not fp_rows.empty:
        fp_rows.to_excel(xl, index=False, sheet_name="false_positives")
    if not fn_rows.empty:
        fn_rows.to_excel(xl, index=False, sheet_name="false_negatives")

print(f"✅ Report saved to {OUT}")
