import pandas as pd
import os

# Load the files
ground_truth_file = "image_pairs_test1.xlsx"
model_predictions_file = "similarities_20251106_190251.xlsx"
output_file = "comparison_results.xlsx"

# Load Excel data
gt_df = pd.read_excel(ground_truth_file)
model_df = pd.read_excel(model_predictions_file)

# Create lookup for unordered pairs
gt_lookup = {}
for _, row in gt_df.iterrows():
    pair = tuple(sorted([row['img1'].lower(), row['img2'].lower()]))
    gt_lookup[pair] = row['true_count']

# Extract file names from paths
model_df['img1'] = model_df['image1_path'].apply(lambda x: os.path.basename(x).lower())
model_df['img2'] = model_df['image2_path'].apply(lambda x: os.path.basename(x).lower())

# Prepare output rows
output_rows = []

for _, row in model_df.iterrows():
    img1 = row['img1']
    img2 = row['img2']
    model_count = row['face_count']

    pair_key = tuple(sorted([img1, img2]))
    true_count = gt_lookup.get(pair_key)

    if true_count is not None:
        diff = model_count - true_count
        if diff > 0:
            error_stat = "false positive"
        elif diff < 0:
            error_stat = "false negative"
        else:
            error_stat = "correct"
        err_mag = abs(diff)
    else:
        error_stat = "not_processed"
        err_mag = None
        true_count = None

    output_rows.append({
        'img1': img1,
        'img2': img2,
        'model_count': model_count,
        'true_count': true_count,
        'error_stat': error_stat,
        'err_mag': err_mag
    })

output_df = pd.DataFrame(output_rows)
output_df.to_excel(output_file, index=False)