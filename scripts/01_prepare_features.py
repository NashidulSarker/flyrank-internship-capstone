"""
01_prepare_features.py - Ingest raw data, assert contract integrity, and save feature vector.
"""

import os
import pandas as pd
from ml_utils import resolve_data_path, build_feature_matrix


def main():
    raw_path = resolve_data_path("data/raw/content_refresh_anonymized.csv")
    print(f"Loading raw dataset from {raw_path}...")
    df = pd.read_csv(raw_path)
    assert len(df) == 30000, f"Expected 30,000 rows, got {len(df)}"
    assert df['content_id'].nunique() == 30000, "Duplicate content_id detected"
    assert df['client_id'].nunique() == 32, "Expected 32 clients"

    print("Building engineered feature matrix X and label y...")
    X, y = build_feature_matrix(df)

    df_prepared = pd.concat([
        df[['content_id', 'client_id']],
        X,
        pd.Series(y, name='is_declining_label')
    ], axis=1)

    out_dir = os.path.join(os.path.dirname(raw_path), "..", "processed")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.abspath(os.path.join(out_dir, "refresh_feature_vector.csv"))
    df_prepared.to_csv(out_file, index=False)
    print(f"[SUCCESS] Saved prepared feature vector ({df_prepared.shape[0]:,} rows x {df_prepared.shape[1]} cols) to {out_file}")


if __name__ == "__main__":
    main()
