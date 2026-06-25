"""
prepare_dataset.py
==================
Converts the Hannousse & Yahiouche (2021) phishing benchmark dataset
into the simple (url, label) CSV format used by train_model.py.

DATA SOURCE
-----------
The original benchmark is hosted on Mendeley Data:
    https://data.mendeley.com/datasets/c2gw7fy2j4/3
    Citation:  Hannousse, A. & Yahiouche, S. (2021).
               Web page phishing detection.
               Engineering Applications of Artificial Intelligence,
               104, 104347. DOI: 10.1016/j.engappai.2021.104347

The file we use is `dataset_B_05_2020.csv`, which contains 11,430 URLs
with 87 pre-extracted features. We only need the URL string and the
class label (the `status` column with values "phishing" or "legitimate").

A public GitHub mirror of the same file is also available:
    https://github.com/Trieuh2/ml-url-phishing-classifier
    (file: datasets/raw_dataset.csv)

USAGE
-----
    python prepare_dataset.py --input dataset_B_05_2020.csv
    python prepare_dataset.py --input raw_dataset.csv --output urls.csv
"""

import argparse
import sys
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Convert Hannousse benchmark CSV to (url, label) format"
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to dataset_B_05_2020.csv or raw_dataset.csv"
    )
    parser.add_argument(
        "--output", default="urls.csv",
        help="Output CSV path (default: urls.csv)"
    )
    args = parser.parse_args()

    print(f"[INFO] Loading {args.input} ...")
    try:
        df = pd.read_csv(args.input)
    except Exception as e:
        print(f"[ERROR] Could not read {args.input}: {e}")
        sys.exit(1)

    # Normalise column names to lowercase so we don't depend on exact case.
    df.columns = [c.strip().lower() for c in df.columns]

    if "url" not in df.columns:
        print("[ERROR] Input CSV does not have a 'url' column.")
        sys.exit(1)
    if "status" not in df.columns:
        print("[ERROR] Input CSV does not have a 'status' column.")
        print("        (This file may not be the Hannousse benchmark.)")
        sys.exit(1)

    # Map the text labels to the integer labels used by train_model.py.
    label_map = {"legitimate": 0, "phishing": 1}
    df["label"] = df["status"].astype(str).str.lower().str.strip().map(label_map)

    # Drop any row that does not have a valid URL or a recognised label.
    df = df.dropna(subset=["url", "label"])
    df["label"] = df["label"].astype(int)

    # Keep only the two columns we need.
    output_df = df[["url", "label"]]
    output_df.to_csv(args.output, index=False)

    # Summary so the student can confirm everything looks right.
    n_phish = int((output_df["label"] == 1).sum())
    n_safe = int((output_df["label"] == 0).sum())
    print(f"[INFO] Saved {len(output_df):,} URLs to {args.output}")
    print(f"       Legitimate (label 0): {n_safe:,}")
    print(f"       Phishing   (label 1): {n_phish:,}")
    print(f"\n[NEXT]  python train_model.py --dataset {args.output}")


if __name__ == "__main__":
    main()
