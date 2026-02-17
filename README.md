# 🌫️ Karachi AQI Forecasting System

An end-to-end **Air Quality Index (AQI) forecasting system** for Karachi that predicts air quality for the next **24, 48, and 72 hours** using time-series machine learning models on a 100% serverless architecture

The project automates data ingestion, feature engineering, model training, versioning, and deployment, and serves real-time predictions through an interactive Streamlit dashboard.

---
# Live App: 

https://karachiaqipredictor10.streamlit.app/

## 🚀 Project Overview

This system continuously fetches hourly air pollution data and processes it through a structured ML pipeline. It:

- Generates advanced time-based and lag features
- Trains multiple ML models
- Automatically selects the best-performing model
- Versions models in a registry
- Serves predictions via a web dashboard

---

## 🏗️ Architecture

### 1️⃣ Data Ingestion Pipeline (Hourly)
- Fetches AQI and pollutant data
- Stores raw + engineered data in Feature Store
- Runs automatically every hour

### 2️⃣ Feature Engineering

Creates advanced features including:

- Time-based features (hour, day, month, weekend flag)
- Cyclical encoding (sin/cos transformations)
- Lag features (1h, 3h, 6h, 12h, 24h)
- Rolling averages
- Rolling standard deviations
- AQI change & percentage change features

### 3️⃣ Training Pipeline (Daily)
- Runs automatically via GitHub Actions
- Trains multiple models (XGBoost, LightGBM, RandomForest)
- Evaluates using RMSE
- Selects best-performing model
- Saves to Model Registry with incremented version

### 4️⃣ Prediction App (Streamlit)
- Loads latest production model
- Aligns features dynamically with model expectations
- Generates real-time AQI forecasts
- Displays interactive charts
- Shows AQI health impact categories

---

## 🤖 Supported Models

| Model | Type |
|-------|------|
| XGBoost | Gradient Boosting |
| LightGBM | Gradient Boosting |
| Random Forest | Ensemble |

---

## 📊 Features Used

The model uses 30+ engineered features including:

| Category | Features |
|----------|----------|
| Pollutants | PM2.5, PM10, NO₂, O₃, CO, SO₂ |
| Time-based | Hour, day, month, cyclical encodings |
| Lag features | AQI and PM2.5 lags |
| Rolling stats | Rolling mean & rolling std |
| Change features | Short-term and long-term AQI changes, percentage change |

---

---

## 📈 Outputs

- AQI forecast for next **24 hours**
- AQI forecast for next **48 hours**
- AQI forecast for next **72 hours**
- Trend visualizations
- Health impact classification (US AQI standard)

---

## 🧠 Key Highlights

- Fully automated ML lifecycle
- Feature Store integration
- Model Registry with version control
- Robust feature alignment logic
- Production-style architecture
- Modular pipeline design
- Modern Streamlit UI

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python |
| Data | Pandas, NumPy |
| ML | Scikit-learn, XGBoost, LightGBM |
| Feature Store & Registry | Hopsworks |
| CI/CD | GitHub Actions |
| Dashboard | Streamlit |
| Visualization | Matplotlib, Plotly |

---

## ⚙️ Requirements & Installation

### 📌 System Requirements

- Python 3.10+ (Recommended: Python 3.12)
- Git
- Internet connection (for data fetching & Hopsworks integration)
- Hopsworks account (Feature Store + Model Registry)
- GitHub account (for CI/CD pipelines)

### 📦 Python Dependencies
```
streamlit
pandas
numpy
scikit-learn
xgboost
lightgbm
hopsworks
matplotlib
plotly
joblib
python-dotenv
```

---

## 🚀 Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/MuhammadHani03/AQI_Project_10P.git
cd AQI_Project_10P
```

### 2️⃣ Create a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4️⃣ Set Up Environment Variables

Create a `.streamlit/secrets.toml` file:
```toml
HOPSWORKS_API_KEY = "your_hopsworks_api_key"
```

Or set as environment variables:
```bash
# Windows
set HOPSWORKS_API_KEY=your_key_here

# macOS / Linux
export HOPSWORKS_API_KEY=your_key_here
```

### 5️⃣ Run the Streamlit App
```bash
streamlit run app/app.py
```

---

## 🔄 Running Pipelines Manually
```bash
# Run the data ingestion pipeline
python src/single_fetch.py

# Run the training pipeline
python src/training_pipeline.py
```

---

## ✅ Verify Installation

If everything is set up correctly:

- ✅ Streamlit dashboard should open in your browser
- ✅ Model should load from Hopsworks
- ✅ Predictions should generate successfully

---

## 👨‍💻 Developed By

**Muhammad Hani**

---

## 🙏 Acknowledgements

- [Open-Meteo](https://open-meteo.com/) / [OpenWeatherMap](https://openweathermap.org/) for AQI data
- [Hopsworks](https://www.hopsworks.ai/) for Feature Store & Model Registry
- [Streamlit](https://streamlit.io/) for the dashboard framework

---

## ⭐ Show Your Support

If you found this project helpful, please consider giving it a **star** ⭐ on GitHub — it means a lot!

---

*Built with ❤️ for a cleaner Karachi 🌿*
