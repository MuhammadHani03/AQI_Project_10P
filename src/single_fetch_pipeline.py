"""Orchestrator for fetch + feature pipelines.

The fetching and feature engineering logic has been split into
`fetch_pipeline.py` and `feature_pipeline.py`. This file keeps the
original CLI behaviour by calling them in sequence.
"""

from fetch_pipeline import fetch_data, upload_raw_to_hopsworks, RAW_FG_NAME, FG_VERSION
from feature_pipeline import create_features_from_raw
import hopsworks
import pandas as pd


if __name__ == "__main__":
    # Step 1: fetch new data and append to raw FG
    df_new_raw = fetch_data()
    upload_raw_to_hopsworks(df_new_raw)

    # Step 2: run feature creation when there are new raw rows.
    # If fetch returned no new rows but raw FG contains rows that the
    # features FG doesn't have (previous run failed during feature upload),
    # detect that and create features for the missing timestamps.
    if df_new_raw is None or df_new_raw.empty:
        print("⚠️ No new raw rows from fetch. Verifying feature FG coverage...")
        try:
            project = hopsworks.login()
            fs = project.get_feature_store()
            raw_fg = fs.get_feature_group(name=RAW_FG_NAME, version=FG_VERSION)
            features_fg = fs.get_feature_group(name="karachi_aqi_features_oct", version=FG_VERSION)

            if raw_fg is None:
                print("⚠️ Raw FG not found — nothing to do.")
            else:
                df_raw = raw_fg.read()
                if df_raw is None or df_raw.empty:
                    print("⚠️ Raw FG empty — nothing to do.")
                else:
                    raw_max = pd.to_datetime(df_raw['time']).max()
                    if features_fg is None:
                        print("⚠️ Features FG not found — creating features for all raw rows.")
                        create_features_from_raw(new_raw_df=df_raw)
                    else:
                        df_feat = features_fg.read()
                        if df_feat is None or df_feat.empty:
                            print("⚠️ Features FG empty — creating features for all raw rows.")
                            create_features_from_raw(new_raw_df=df_raw)
                        else:
                            feat_max = pd.to_datetime(df_feat['time']).max()
                            if raw_max > feat_max:
                                print("⚠️ Raw FG has newer rows than features FG — creating missing features.")
                                # Pass only the new raw slice to avoid recomputing everything
                                df_new_slice = df_raw[pd.to_datetime(df_raw['time']) > feat_max].copy()
                                create_features_from_raw(new_raw_df=df_new_slice)
                            else:
                                print("✅ Features FG is up-to-date with raw FG. Nothing to do.")
        except Exception as e:
            print(f"⚠️ Could not verify FG coverage: {e}\nSkipping feature engineering.")
    else:
        create_features_from_raw(new_raw_df=df_new_raw)
