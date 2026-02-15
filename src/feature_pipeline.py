import pandas as pd
import numpy as np
import hopsworks

from fetch_pipeline import RAW_FG_NAME, FG_VERSION


def feature_engineering(df):
    """
    Feature engineering for AQI prediction.
    Adds time features, cyclical encoding, lags, rolling, interactions,
    ratio features, and multi-horizon future targets.
    """
    if df.empty:
        return df
    df = df.copy()

    # Normalize time and sort once
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)

    # Rename raw columns in one step (keeps original df usable for vector ops)
    df = df.rename(columns={
        "temperature_2m": "temp",
        "relative_humidity_2m": "humidity",
        "surface_pressure": "pressure",
        "wind_speed_10m": "wind_speed",
        "cloud_cover": "clouds",
        "carbon_monoxide": "co",
        "nitrogen_dioxide": "no2",
        "sulphur_dioxide": "so2",
        "ozone": "o3"
    })

    # Build all new columns in a dict, then concat once to avoid fragmentation
    new_cols = {}

    # Time features
    new_cols['hour'] = df['time'].dt.hour
    new_cols['day_of_week'] = df['time'].dt.dayofweek
    new_cols['day'] = df['time'].dt.day
    new_cols['month'] = df['time'].dt.month
    new_cols['is_weekend'] = df['time'].dt.dayofweek.isin([5,6]).astype(int)
    new_cols['hour_sin'] = np.sin(2 * np.pi * new_cols['hour'] / 24)
    new_cols['hour_cos'] = np.cos(2 * np.pi * new_cols['hour'] / 24)

    # Lags for AQI
    lag_hours = [1, 3, 6, 12, 24]
    for lag in lag_hours:
        new_cols[f'aqi_lag_{lag}h'] = df['us_aqi'].shift(lag)

    # PM2.5 lags
    new_cols['pm25_lag_1h'] = df['pm2_5'].shift(1)
    new_cols['pm25_lag_24h'] = df['pm2_5'].shift(24)

    # Rolling features (use shifted series for look-ahead safety)
    shifted_aqi = df['us_aqi'].shift(1)
    rolling_windows = [3, 6, 12, 24]
    for window in rolling_windows:
        new_cols[f'aqi_rolling_{window}h'] = shifted_aqi.rolling(window).mean()

    new_cols['aqi_std_24h'] = shifted_aqi.rolling(24).std()
    new_cols['pm25_rolling_6h'] = df['pm2_5'].shift(1).rolling(6).mean()
    new_cols['pm25_rolling_24h'] = df['pm2_5'].shift(1).rolling(24).mean()

    # Change features
    new_cols['aqi_change_1h'] = df['us_aqi'] - df['us_aqi'].shift(1)
    new_cols['aqi_change_3h'] = df['us_aqi'] - df['us_aqi'].shift(3)
    new_cols['aqi_change_6h'] = df['us_aqi'] - df['us_aqi'].shift(6)
    new_cols['aqi_change_24h'] = df['us_aqi'] - df['us_aqi'].shift(24)
    new_cols['aqi_pct_change_1h'] = new_cols['aqi_change_1h'] / (df['us_aqi'].shift(1) + 1e-6)
    new_cols['aqi_pct_change_24h'] = new_cols['aqi_change_24h'] / (df['us_aqi'].shift(24) + 1e-6)

    # Future targets
    new_cols['aqi_next_24h'] = df['us_aqi'].shift(-24)
    new_cols['aqi_next_48h'] = df['us_aqi'].shift(-48)
    new_cols['aqi_next_72h'] = df['us_aqi'].shift(-72)

    # Keep core columns (pollutants, raw aqi)
    keep_cols = ['pm2_5', 'pm10', 'no2', 'o3', 'co', 'so2', 'us_aqi']

    # Concatenate once
    new_df = pd.concat([df[['time']].reset_index(drop=True), pd.DataFrame(new_cols, index=df.index)], axis=1)
    for c in keep_cols:
        if c in df.columns:
            new_df[c] = df[c].values

    # Final feature list (fixed target names)
    feature_cols = [
        'time',
        'hour', 'day_of_week', 'day', 'month', 'is_weekend',
        'hour_sin', 'hour_cos',
        'temp', 'humidity', 'pressure', 'wind_speed', 'clouds',
        'pm2_5', 'pm10', 'no2', 'o3', 'co', 'so2',
        'aqi_lag_1h', 'aqi_lag_3h', 'aqi_lag_6h', 'aqi_lag_12h', 'aqi_lag_24h',
        'pm25_lag_1h', 'pm25_lag_24h', 'us_aqi',
        'aqi_rolling_3h', 'aqi_rolling_6h', 'aqi_rolling_12h', 'aqi_rolling_24h',
        'aqi_std_24h', 'pm25_rolling_6h', 'pm25_rolling_24h',
        'aqi_change_1h', 'aqi_change_3h', 'aqi_change_6h', 'aqi_change_24h',
        'aqi_pct_change_1h', 'aqi_pct_change_24h',
        'aqi_next_24h', 'aqi_next_48h', 'aqi_next_72h'
    ]

    # Subset (only keep columns that exist to avoid KeyError)
    feature_cols = [c for c in feature_cols if c in new_df.columns]
    result = new_df[feature_cols].copy()
    return result


def create_features_from_raw(new_raw_df=None):
    """Create features from raw data.

    If `new_raw_df` is provided, only compute features for timestamps that
    are new compared to the existing features FG, using a historical window
    sufficient for lags/rolling (24h by default).
    """
    project = hopsworks.login()
    fs = project.get_feature_store()

    # --- Read raw data ---
    raw_fg = fs.get_feature_group(name=RAW_FG_NAME, version=FG_VERSION)
    df_raw_full = raw_fg.read()
    if df_raw_full.empty:
        print("⚠️ No raw data found.")
        return

    # If no new_raw_df provided, behave as before (recompute all features)
    if new_raw_df is None or new_raw_df.empty:
        df_features = feature_engineering(df_raw_full)
    else:
        # Determine required historical window (max lag / rolling used in feature_engineering)
        history_hours = 24
        new_min = pd.to_datetime(new_raw_df["time"]).min()
        new_max = pd.to_datetime(new_raw_df["time"]).max()
        if new_min.tzinfo is None:
            new_min = new_min.tz_localize("UTC")
        if new_max.tzinfo is None:
            new_max = new_max.tz_localize("UTC")

        start_needed = new_min - pd.Timedelta(hours=history_hours)

        # Subset raw data to the needed window (safe even if df_raw_full small)
        df_subset = df_raw_full.copy()
        df_subset["time"] = pd.to_datetime(df_subset["time"]).dt.tz_convert("UTC")
        df_subset = df_subset[df_subset["time"] >= start_needed].sort_values("time").reset_index(drop=True)

        # Compute features on the subset, then keep only rows that correspond to new_raw_df times
        df_features_all = feature_engineering(df_subset)

        # Only keep feature rows within the new timestamps range
        df_features = df_features_all[(df_features_all["time"] >= new_min) & (df_features_all["time"] <= new_max)].copy()

    # --- Upload features (append only new rows if FG exists) ---
    try:
        fg = fs.get_feature_group(name="karachi_aqi_features_oct", version=FG_VERSION)
        if fg is None:
            raise Exception("Features FG not found, creating...")
        print(f"✅ Feature group 'karachi_aqi_features_oct' found.")
        # read existing to detect already-uploaded times
        existing = fg.read()
        if not existing.empty:
            existing_max = pd.to_datetime(existing["time"]).max()
            if existing_max.tzinfo is None:
                existing_max = existing_max.tz_localize("UTC")
            df_to_upload = df_features[pd.to_datetime(df_features["time"]) > existing_max]
        else:
            df_to_upload = df_features
    except Exception:
        fg = fs.create_feature_group(
            name="karachi_aqi_features",
            version=FG_VERSION,
            description="Karachi AQI engineered features",
            primary_key=["time"],
            event_time="time",
            online_enabled=False
        )
        print(f"✅ Feature group 'karachi_aqi_features_oct' created.")
        df_to_upload = df_features

    if df_to_upload is None or df_to_upload.empty:
        print("⚠️ No new feature rows to upload.")
        return

    # Attempt to insert with retries to handle transient connection failures
    import time
    from requests.exceptions import ConnectionError as RequestsConnectionError
    from urllib3.exceptions import ProtocolError
    from http.client import RemoteDisconnected as HTTPRemoteDisconnected

    max_retries = 5
    for attempt in range(1, max_retries + 1):
        try:
            fg.insert(df_to_upload, write_options={"offline": True, "wait_for_job": True})
            print(f"🚀 Uploaded {len(df_to_upload)} new feature rows to Hopsworks.")
            break
        except (RequestsConnectionError, ProtocolError, HTTPRemoteDisconnected) as e:
            print(f"⚠️ Hopsworks connection error on attempt {attempt}: {e}")
            if attempt == max_retries:
                print("❌ Max retries reached — aborting upload.")
                raise
            sleep_time = 2 ** attempt
            print(f"   Retrying in {sleep_time}s...")
            time.sleep(sleep_time)
        except Exception as e:
            print(f"❌ Upload failed: {e}")
            raise
