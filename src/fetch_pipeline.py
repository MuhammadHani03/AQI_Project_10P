import json
from pathlib import Path
import time

import pandas as pd
from dotenv import load_dotenv
import hopsworks

from open_meteo_client_api import fetch_weather, fetch_air_quality

load_dotenv()

# -----------------------------
# CONFIG (for fetch/upload)
# -----------------------------
RAW_FG_NAME = "karachi_aqi_raw_data"
FG_VERSION = 1
STATE_PATH = Path("data/state.json")


def get_last_timestamp_from_hopsworks():
    """Try to read the last `time` value from the raw FG in Hopsworks.
    Returns a tz-aware UTC pandas.Timestamp or None if FG not found / unreachable.
    """
    try:
        project = hopsworks.login()
        fs = project.get_feature_store()
        fg = fs.get_feature_group(name=RAW_FG_NAME, version=FG_VERSION)
        if fg is None:
            return None
        df = fg.read()
        if df.empty:
            return None
        ts = pd.to_datetime(df["time"]).max()
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return ts
    except Exception:
        return None


def read_state():
    """Read last timestamp preferring Hopsworks raw FG, fall back to JSON state.

    This ensures the pipeline uses the authoritative source (raw FG) when
    available, avoiding stale local `state.json` issues.
    """
    # Prefer hopsworks FG timestamp when available
    ts = get_last_timestamp_from_hopsworks()
    if ts is not None:
        return ts

    # Fall back to local JSON state
    if STATE_PATH.exists():
        with open(STATE_PATH, "r") as f:
            state = json.load(f)
            ts = pd.to_datetime(state.get("last_timestamp"))
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            return ts
    return None


def update_state(last_timestamp):
    """Save last timestamp to JSON state."""
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump({"last_timestamp": str(last_timestamp)}, f)


def fetch_data():
    """Fetch new data from Open-Meteo and avoid future timestamps."""
    last_ts = read_state()
    start_ts = (last_ts) if last_ts else pd.Timestamp("2025-01-01 00:00:00", tz="UTC")
    
    end_ts = pd.Timestamp.now(tz="UTC").floor("h")

    if start_ts >= end_ts:
        print("⚠️ No new data to fetch.")
        return pd.DataFrame()

    df_list = []
    chunk_start = start_ts
    while chunk_start < end_ts:
        chunk_end = min(chunk_start + pd.DateOffset(months=1) - pd.Timedelta(hours=1), end_ts)
        start_date_str = chunk_start.strftime("%Y-%m-%d")
        end_date_str = chunk_end.strftime("%Y-%m-%d")
        print(f"Fetching chunk: {start_date_str} → {end_date_str}")

        # Fetch weather & AQ data with retry (per-chunk)
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                weather_df = fetch_weather(start_date_str, end_date_str)
                air_df = fetch_air_quality(start_date_str, end_date_str)
                break
            except Exception as e:
                print(f"Fetch attempt {attempt} failed: {e}")
                if attempt == max_attempts:
                    raise
                time.sleep(2 * attempt)

        chunk_df = pd.merge(weather_df, air_df, on="time", how="inner")

        # Ensure 'time' column is tz-aware UTC
        chunk_df['time'] = pd.to_datetime(chunk_df['time']).dt.tz_localize('UTC', ambiguous='NaT', nonexistent='shift_forward')

        # Remove already fetched and future timestamps
        if last_ts is not None:
            chunk_df = chunk_df[chunk_df["time"] > last_ts]
        chunk_df = chunk_df[chunk_df["time"] <= end_ts]

        df_list.append(chunk_df)
        chunk_start = chunk_end + pd.Timedelta(hours=1)

    if df_list:
        df = pd.concat(df_list).sort_values("time").reset_index(drop=True)
        update_state(df["time"].max())
        print(f"✅ Fetched {len(df)} new rows | Last timestamp: {df['time'].max()}")
        return df
    else:
        print("⚠️ No new data fetched.")
        return pd.DataFrame()


def upload_raw_to_hopsworks(df: pd.DataFrame):
    if df.empty:
        print("⚠️ No new raw data to upload.")
        return

    project = hopsworks.login()
    fs = project.get_feature_store()

    try:
        fg = fs.get_feature_group(name=RAW_FG_NAME, version=FG_VERSION)
        if fg is None:
            raise Exception("Raw FG not found, creating...")
        print(f"✅ Raw feature group '{RAW_FG_NAME}' found.")
    except:
        fg = fs.create_feature_group(
            name=RAW_FG_NAME,
            version=FG_VERSION,
            description="Karachi AQI raw data",
            primary_key=["time"],
            event_time="time",
            online_enabled=False
        )
        print(f"✅ Raw feature group '{RAW_FG_NAME}' created.")

    fg.insert(df, write_options={"offline": True, "wait_for_job": True})
    print(f"🚀 Uploaded {len(df)} raw rows to Hopsworks.")
