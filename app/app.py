import streamlit as st
import hopsworks
import pandas as pd
import numpy as np
import joblib
import os
import pytz
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px


# CONFIG
# -----------------------------
FG_NAME = "karachi_aqi_features_oct"
FG_VERSION = 1
TARGET_COLS = ["aqi_next_24h", "aqi_next_48h", "aqi_next_72h"]
TIMESTAMP_COL = "time"


# LOAD MODEL & INFO FROM HOPSWORKS
# -----------------------------
@st.cache_resource
def load_model_and_info():
    project = hopsworks.login(api_key_value=st.secrets["HOPSWORKS_API_KEY"])
    mr = project.get_model_registry()

    # Base model name
    base_name = "1_year_champ"

    try:
        # Get the best model automatically based on RMSE (lower is better)
        best_model_obj = mr.get_best_model(name=base_name, metric="rmse", direction="min")
    except Exception:
        # Fallback to local model if nothing found in registry
        local_model_path = os.path.join("aqi_model_dir", "model.pkl")
        if os.path.exists(local_model_path):
            loaded_model = joblib.load(local_model_path)

            model_info = {
                "name": "Best_Model",
                "version": "local",
                "metrics": {
                    "RMSE": "N/A",
                    "MAE": "N/A",
                    "R2": "N/A",
                },
            }
            return loaded_model, model_info

        raise RuntimeError(
            f"No models found in the registry matching '{base_name}'.\n"
            "Ensure models were registered and the API key/project have access."
        )

    # Download and load the best model
    model_dir = best_model_obj.download()
    model_path = os.path.join(model_dir, "model.pkl")
    loaded_model = joblib.load(model_path)

    metrics_dict = getattr(best_model_obj, "training_metrics", {}) or {}
    metrics = {
        "RMSE": metrics_dict.get("rmse", "N/A"),
        "MAE": metrics_dict.get("mae", "N/A"),
        "R2": metrics_dict.get("r2", "N/A"),
    }

    model_info = {
        "name": best_model_obj.name,
        "version": best_model_obj.version,
        "metrics": metrics,
    }

    return loaded_model, model_info

# Load model
model, model_info = load_model_and_info()


# FEATURE STORE
# -----------------------------
@st.cache_resource
def get_hopsworks_feature_store():
    project = hopsworks.login(api_key_value=st.secrets["HOPSWORKS_API_KEY"])
    fs = project.get_feature_store()
    return fs

fs = get_hopsworks_feature_store()


# GET LATEST FEATURES
# -----------------------------
def get_latest_features():
    fg = fs.get_feature_group(name=FG_NAME, version=FG_VERSION)
    df = fg.read()
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], unit="ns")
    df = df.sort_values(TIMESTAMP_COL, ascending=False)
    latest_row = df.iloc[0]

    input_features = latest_row.drop(TARGET_COLS + [TIMESTAMP_COL])
    features_df = pd.DataFrame([input_features.values], columns=input_features.index)

    return features_df, latest_row


# MAKE PREDICTIONS
# -----------------------------
def make_local_prediction(features_df):
    predictions = model.predict(features_df.to_numpy())
    return predictions[0]


# AQI CATEGORY & COLOR
# -----------------------------
def get_aqi_category(aqi):
    if aqi <= 50: return "Good 😊"
    elif aqi <= 100: return "Moderate 😐"
    elif aqi <= 150: return "Unhealthy (Sensitive) 😷"
    elif aqi <= 200: return "Unhealthy ⚠️"
    elif aqi <= 300: return "Very Unhealthy 🚨"
    else: return "Hazardous ☠️"

def get_aqi_color(aqi):
    if aqi <= 50: return "#4caf50"
    elif aqi <= 100: return "#ffeb3b"
    elif aqi <= 150: return "#ff9800"
    elif aqi <= 200: return "#f44336"
    elif aqi <= 300: return "#9c27b0"
    else: return "#000000"



# CUSTOM CSS
# -----------------------------
st.markdown(
    """
    <style>
    body {
        background-color: #ffffff;
        color: #1a237e;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stMain {
        background-color: #ffffff;
    }
    [data-testid="stAppViewContainer"] {
        background-color: #ffffff;
    }
    [data-testid="stVerticalBlock"] {
        background: transparent;
    }
    .stHeader {
        font-size: 3rem;
        font-weight: bold;
        color: #0d47a1;
        text-align: center;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1);
    }
    .stMetric > div {
        background-color: rgba(245, 245, 245, 0.9);
        border-radius: 15px;
        padding: 15px;
        box-shadow: 0 4px 15px rgba(13, 71, 161, 0.1);
        border-left: 4px solid #1976d2;
    }
    .stDataFrame {
        border-radius: 10px;
        box-shadow: 0 4px 10px rgba(13, 71, 161, 0.1);
        background-color: rgba(245, 245, 245, 0.9);
    }
    div.stButton > button:first-child {
        background-color: #1976d2;
        color: white;
        font-weight: bold;
        padding: 0.7rem 1.5rem;
        border-radius: 8px;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 10px rgba(25, 118, 210, 0.2);
    }
    div.stButton > button:first-child:hover {
        background-color: #0d47a1;
        cursor: pointer;
        box-shadow: 0 6px 15px rgba(13, 71, 161, 0.3);
        transform: translateY(-2px);
    }
    h2, h3 {
        color: #0d47a1;
        font-weight: 600;
    }
    hr {
        background: linear-gradient(to right, rgba(13, 71, 161, 0.2), rgba(13, 71, 161, 0.6), rgba(13, 71, 161, 0.2));
        border: 0;
        height: 2px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# STREAMLIT SIDEBAR
# -----------------------------
st.sidebar.header("🔹 Model Information")
st.sidebar.markdown(f"""
**Active Model:** {model_info['name']}  
**Version:** {model_info['version']}  


""")

st.sidebar.divider()

st.sidebar.header("🔹 About")
st.sidebar.markdown(f"""

This model predicts the **Air Quality Index (AQI)** for Karachi for the next 24, 48, and 72 hours using historical features. It helps track pollution levels and plan activities accordingly.
""")

st.sidebar.markdown(
    """
    <hr>
    <p style='text-align: center; font-size: 13px;'>
        © 2026 AQI Predictor <br>
        Developed by Muhammad Hani
    </p>
    """,
    unsafe_allow_html=True
)

# MAIN UI
# -----------------------------
st.title("🌆 Karachi AQI Forecast For Next 3 Days")

# Fetch latest features
features_df, latest_row = get_latest_features()

# Convert timestamp to PKT for display
pkt_tz = pytz.timezone("Asia/Karachi")
latest_row[TIMESTAMP_COL] = latest_row[TIMESTAMP_COL].tz_convert(pkt_tz)

# Select only the columns you want to display
cols_to_show = [TIMESTAMP_COL, "us_aqi", "pm2_5", "pm10"]
display_row = latest_row[cols_to_show]

# Format timestamp nicely
display_row[TIMESTAMP_COL] = display_row[TIMESTAMP_COL].strftime("%Y-%m-%d %H:%M:%S %Z")

# Convert to dataframe for Streamlit display
display_df = display_row.to_frame("Value")

st.subheader("Latest Features (Used for Prediction)")
st.dataframe(display_df.astype(str))


def get_hist_and_current_aqi_24h():
    fg = fs.get_feature_group(name=FG_NAME, version=FG_VERSION)
    df = fg.read()
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], unit="ns")
    df = df.sort_values(TIMESTAMP_COL)

    # Latest timestamp (current)
    latest_time = df[TIMESTAMP_COL].max()

    # Past 24h-spaced timestamps before the latest
    past_times = [latest_time - pd.Timedelta(hours=h) for h in [72, 48, 24]]

    # Fetch historical points
    df_hist_points = df[df[TIMESTAMP_COL].isin(past_times)][[TIMESTAMP_COL, "us_aqi"]]

    # Add current/latest AQI as its own point
    df_current = df[df[TIMESTAMP_COL] == latest_time][[TIMESTAMP_COL, "us_aqi"]]

    # Combine historical + current
    df_hist_combined = pd.concat([df_hist_points, df_current], ignore_index=True)

    # Convert to PKT timezone
    pkt_tz = pytz.timezone("Asia/Karachi")
    df_hist_combined[TIMESTAMP_COL] = df_hist_combined[TIMESTAMP_COL].dt.tz_convert(pkt_tz)

    return df_hist_combined



def combine_hist_current_and_pred_24h(df_hist_current, predictions):
    # Latest timestamp from historical/current
    last_time = df_hist_current[TIMESTAMP_COL].max()

    # Future prediction timestamps
    horizons = [24, 48, 72]
    pred_times = [last_time + pd.Timedelta(hours=h) for h in horizons]

    df_pred = pd.DataFrame({
        TIMESTAMP_COL: pred_times,
        "AQI": predictions,
        "Type": "Forecast"
    })

    df_hist_plot = df_hist_current.copy()
    df_hist_plot.rename(columns={"us_aqi": "AQI"}, inplace=True)
    df_hist_plot["Type"] = "Historical"

    # Combine historical/current + forecast
    df_combined = pd.concat([df_hist_plot, df_pred], ignore_index=True)
    return df_combined

# AQI Prediction Button
# -----------------------------
if st.button("Predict AQI (Next 3 Days)"):
    with st.spinner("Running prediction..."):
        predictions = make_local_prediction(features_df)

    st.divider()
    st.subheader("📊 3-Day AQI Forecast")

    horizons = [24, 48, 72]
    cols = st.columns(3)

    for i, hrs in enumerate(horizons):
        with cols[i]:
            st.metric(
                label=f"{hrs}h",
                value=round(predictions[i], 1),
                delta=get_aqi_category(predictions[i])
            )

    # Create chart data
    chart_data = {
        "Horizon": ["24h", "48h", "72h"],
        "AQI": predictions,
        "Category": [get_aqi_category(a) for a in predictions],
        "Color": [get_aqi_color(a) for a in predictions]
    }
    df_predictions = pd.DataFrame(chart_data)

    st.divider()

    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs([
        "📊 Bar Chart",
        "🔢 Metrics Table",
        "💡 Health Suggestions"
    ])

    # Tab 1: Bar Chart (Plotly)
    with tab1:
        st.subheader("AQI Forecast - Bar Chart")
        fig_bar = go.Figure(data=[
            go.Bar(
                x=df_predictions["Horizon"],
                y=df_predictions["AQI"],
                marker=dict(
                    color=df_predictions["Color"],
                    line=dict(color='#0d47a1', width=2)
                ),
                text=[f"{round(v, 1)}" for v in predictions],
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>AQI: %{y:.1f}<extra></extra>',
                name="AQI"
            )
        ])
        fig_bar.update_layout(
            title="AQI Forecast (Next 3 Days)",
            xaxis_title="Forecast Horizon",
            yaxis_title="AQI Value",
            showlegend=False,
            height=400,
            template="plotly_white",
            font=dict(family="Segoe UI", size=12, color="#0d47a1"),
            plot_bgcolor="rgba(240, 248, 255, 0.5)",
            paper_bgcolor="rgba(255, 255, 255, 0.9)"
        )
        st.plotly_chart(fig_bar, use_container_width=True)


    # Tab 5: Metrics Table
    with tab2:
        st.subheader("Detailed Forecast Metrics")
        detailed_metrics = pd.DataFrame({
            "Horizon": ["24 Hours", "48 Hours", "72 Hours"],
            "AQI Value": [f"{round(p, 2)}" for p in predictions],
            "Category": df_predictions["Category"],
            "Health Status": ["✅ Good" if p <= 50 else "⚠️ Moderate" if p <= 100 else "🚨 Unhealthy" if p <= 150 else "❌ Very Unhealthy" for p in predictions]
        })
        st.dataframe(detailed_metrics, use_container_width=True, hide_index=True)

        # Tab 3: Health Suggestions
    with tab3:
        st.subheader("💡 AQI-Based Health Recommendations")

        def get_health_suggestion(aqi):
            if aqi <= 50:
                return "Air quality is good. No precautions needed. ✅"
            elif aqi <= 100:
                return "Moderate air quality. Sensitive individuals may consider wearing a light mask. ⚠️"
            elif aqi <= 150:
                return "Unhealthy for sensitive groups. Wear a mask outdoors and limit prolonged exposure. 🚨"
            elif aqi <= 200:
             return "Unhealthy! Wear a proper mask, avoid outdoor activities. ❌"
            elif aqi <= 300:
                return "Very unhealthy! Stay indoors, use air purifiers if possible, wear high-quality mask if going out. 🚨"
            else:
                return "Hazardous! Avoid all outdoor activities, keep windows closed, use proper masks indoors if needed. ☠️"

    # Create dataframe for suggestions
        suggestions_df = pd.DataFrame({
        "Horizon": ["24 Hours", "48 Hours", "72 Hours"],
        "Predicted AQI": [round(p, 1) for p in predictions],
        "Suggestion": [get_health_suggestion(p) for p in predictions]
     })

        st.dataframe(suggestions_df, use_container_width=True, hide_index=True)


# Line Chart: Past + Current + Forecast (24h intervals)
# -----------------------------
    st.divider()
    st.subheader("📈 AQI Timeline (Past + Current + Forecast, 24h intervals)")

# Get historical + current AQI
    df_hist_current = get_hist_and_current_aqi_24h()

# Future prediction timestamps
    last_time = df_hist_current[TIMESTAMP_COL].max()
    horizons = [24, 48, 72]
    pred_times = [last_time + pd.Timedelta(hours=h) for h in horizons]

    df_future = pd.DataFrame({
        TIMESTAMP_COL: pred_times,
        "AQI": predictions
})

   # Combine past + current + forecast into one dataframe
    df_line = pd.concat([
    df_hist_current.rename(columns={"us_aqi": "AQI"}),  # past + current
    df_future
    ], ignore_index=True)

    # Convert timestamps to PKT
    pkt_tz = pytz.timezone("Asia/Karachi")
    df_line[TIMESTAMP_COL] = pd.to_datetime(df_line[TIMESTAMP_COL], errors="coerce")
    df_line[TIMESTAMP_COL] = df_line[TIMESTAMP_COL].dt.tz_convert(pkt_tz)

    # Determine which points are future
    df_line["is_forecast"] = [False]*len(df_hist_current) + [True]*len(df_future)

    # Single line trace
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(
    x=df_line[TIMESTAMP_COL],
    y=df_line["AQI"],
    mode="lines+markers",
    line=dict(color="#0d47a1", width=3),  # continuous line
    marker=dict(
        size=8,
        color=np.where(df_line["is_forecast"], "#42a5f5", "#0d47a1")  # past/current = dark blue, future = light blue
    ),
    name="AQI"
))

    fig_line.update_layout(
    template="plotly_white",
    font=dict(family="Segoe UI", size=12, color="#0d47a1"),
    height=500,
    xaxis_title="Time (PKT)",
    yaxis_title="AQI",
    showlegend=False
)

    st.plotly_chart(fig_line, use_container_width=True)
