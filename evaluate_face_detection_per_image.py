# Compare per-image face counts: model output (face_count) vs manual GT (true_faces).

import os
import math
import pandas as pd

# -------- CONFIG --------
GT_FILE       = r"face_recognition_ground_truth1.xlsx"      # columns: image_id, true_faces
RESULTS_FILE  = r"image_results_20251106_190251.xlsx"       # columns: image_path, face_count (from main code)
OUT_XLSX      = r"retinaface_detection_report_from_results.xlsx"
# ------------------------

def stem(path: str) -> str:
    """Return filename without extension, lowercased."""
    b = os.path.basename(str(path))
    return os.path.splitext(b)[0].strip().lower()

def load_gt(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    if not {"image_id", "true_faces"}.issubset(df.columns):
        raise ValueError("GT must contain columns: image_id, true_faces")
    df = df.copy()
    df["stem"] = df["image_id"].astype(str).map(stem)
    df["true_faces"] = pd.to_numeric(df["true_faces"], errors="coerce").fillna(0).astype(int)
    return df[["stem", "image_id", "true_faces"]]

def load_results(path: str) -> pd.DataFrame:
    df = pd.read_excel(path) if path.lower().endswith((".xlsx", ".xls")) else pd.read_parquet(path)
    if not {"image_path", "face_count"}.issubset(df.columns):
        raise ValueError("RESULTS must contain columns: image_path, face_count")
    df = df.copy()
    df["stem"] = df["image_path"].astype(str).map(stem)
    df["face_count"] = pd.to_numeric(df["face_count"], errors="coerce").fillna(0).astype(int)
    return df[["stem", "image_path", "face_count"]]

def main():
    gt = load_gt(GT_FILE)
    res = load_results(RESULTS_FILE)

    # Join by stem (filename without extension)
    merged = gt.merge(res, on="stem", how="left", suffixes=("_gt", "_model"))

    # Coverage: how many GT rows appeared in the model results
    matched = merged["image_path"].notna().sum()
    coverage = matched / len(merged) if len(merged) else 0.0

    # For GT rows not found in model results, treat face_count as 0
    merged["face_count"] = merged["face_count"].fillna(0).astype(int)

    # Per-image contributions
    merged["TP_i"] = (merged[["true_faces", "face_count"]].min(axis=1)).astype(int)
    merged["FP_i"] = (merged["face_count"] - merged["true_faces"]).clip(lower=0).astype(int)
    merged["FN_i"] = (merged["true_faces"] - merged["face_count"]).clip(lower=0).astype(int)

    # Aggregate totals
    TP = int(merged["TP_i"].sum())
    FP = int(merged["FP_i"].sum())
    FN = int(merged["FN_i"].sum())

    # Count-based metrics
    accuracy = TP / (TP + FN) if (TP + FN) > 0 else 0.0   # recall of face counts
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = accuracy
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Optional error measures
    mae = float((merged["face_count"] - merged["true_faces"]).abs().mean())
    rmse = math.sqrt(float(((merged["face_count"] - merged["true_faces"])**2).mean()))

    summary = pd.DataFrame([{
        "detector": "retinaface (from results file)",
        "images_in_gt": len(gt),
        "images_matched": matched,
        "coverage": coverage,
        "TP": TP, "FP": FP, "FN": FN,
        "Accuracy(count-recall)": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "MAE": mae,
        "RMSE": rmse
    }])

    # Rows from GT not found in results
    not_found = merged[merged["image_path"].isna()][["image_id", "true_faces", "stem"]].copy()

    # per-image table
    per_image = merged[[
        "image_id", "image_path", "true_faces", "face_count", "TP_i", "FP_i", "FN_i"
    ]].sort_values("image_id")

    with pd.ExcelWriter(OUT_XLSX) as w:
        summary.to_excel(w, sheet_name="summary", index=False)
        per_image.to_excel(w, sheet_name="per_image", index=False)
        if not not_found.empty:
            not_found.to_excel(w, sheet_name="not_found_in_results", index=False)

    print("\n=== RetinaFace — Detection Metrics (from existing results) ===")
    print(summary.to_string(index=False))
    print(f"\n✅ Saved: {OUT_XLSX}")

if __name__ == "__main__":
    main()
