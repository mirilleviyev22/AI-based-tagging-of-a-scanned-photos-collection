import pandas as pd
import os
from pathlib import Path


def convert_parquet_to_excel(timestamp=None):

    # Find parquet files
    if timestamp:
        results_file = f"image_results_{timestamp}.parquet"
        similarities_file = f"similarities_{timestamp}.parquet"
    else:
        # Find most recent files
        results_files = list(Path(".").glob("image_results_*.parquet"))
        similarities_files = list(Path(".").glob("similarities_*.parquet"))

        if not results_files:
            print("No parquet files found!")
            return

        # Get most recent
        results_file = max(results_files, key=lambda f: f.stat().st_mtime)
        similarities_file = max(similarities_files, key=lambda f: f.stat().st_mtime)

    # Convert results file
    if os.path.exists(results_file):
        print(f"Converting {results_file}...")
        df_results = pd.read_parquet(results_file)

        # Create Excel filename
        excel_results = str(results_file).replace('.parquet', '.xlsx')
        df_results.to_excel(excel_results, index=False)
        print(f"✅ Saved: {excel_results}")
    else:
        print(f"❌ File not found: {results_file}")

    # Convert similarities file
    if os.path.exists(similarities_file):
        print(f"Converting {similarities_file}...")
        df_similarities = pd.read_parquet(similarities_file)

        excel_similarities = str(similarities_file).replace('.parquet', '.xlsx')
        df_similarities.to_excel(excel_similarities, index=False)
        print(f"✅ Saved: {excel_similarities}")
    else:
        print(f"❌ File not found: {similarities_file}")


# Usage:
if __name__ == "__main__":
    # Convert most recent files
    convert_parquet_to_excel()

