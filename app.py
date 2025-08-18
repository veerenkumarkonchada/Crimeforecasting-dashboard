# app.py
import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from math import sqrt

# --- Optional imports (graceful fallbacks) ---
PROPHET_AVAILABLE = True
try:
    from prophet import Prophet
except Exception:
    PROPHET_AVAILABLE = False

LSTM_AVAILABLE = True
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.preprocessing import MinMaxScaler
except Exception:
    LSTM_AVAILABLE = False


# ==============================
# Streamlit Page Setup
# ==============================
st.set_page_config(page_title="Crime Forecasting Dashboard", layout="wide")
st.title("📊 Hybrid Crime Forecasting Dashboard")
st.markdown("Forecast the next **N** years using a **hybrid engine (ARIMA, Prophet, LSTM)** with automatic model selection.")

# ==============================
# Config / Constants
# ==============================
FILE_PATH = "AI PROJECT(2).xlsx"   # keep your filename
TEST_HORIZON = 3                   # last K years reserved for validation
FORECAST_YEARS_DEFAULT = 6
MIN_POINTS_FOR_ARIMA = 5
MIN_POINTS_FOR_PROPHET = 6
MIN_POINTS_FOR_LSTM = 10
LSTM_WINDOW = 3                    # sliding window length
ANOMALY_Z = 2.0                    # z-score threshold for highlighting anomalies
SEED = 42
np.random.seed(SEED)

# ==============================
# Utils
# ==============================
def rmse(y_true, y_pred):
    return sqrt(mean_squared_error(y_true, y_pred))

def safe_mape(y_true, y_pred):
    # avoid division by zero; add small epsilon
    eps = 1e-8
    return mean_absolute_percentage_error(np.array(y_true) + eps, np.array(y_pred) + eps)

def split_series(years, values, test_horizon=TEST_HORIZON):
    if len(values) <= test_horizon:
        return (years, values, np.array([]), np.array([]))
    return (years[:-test_horizon], values[:-test_horizon],
            years[-test_horizon:], values[-test_horizon:])

def ci_from_residuals(pred, residuals, z=1.96):
    # approximate CI using residual std-dev (for LSTM)
    sigma = np.std(residuals) if len(residuals) > 1 else 0.0
    lower = pred - z * sigma
    upper = pred + z * sigma
    return lower, upper

def zscore_anomalies(series, threshold=2.0):
    if len(series) < 3 or np.std(series) == 0:
        return np.array([False]*len(series))
    z = (series - np.mean(series)) / np.std(series)
    return np.abs(z) >= threshold

# ==============================
# Data Load
# ==============================
@st.cache_data(show_spinner=False)
def load_data(path):
    df_local = pd.read_excel(path, header=0)
    return df_local

try:
    df = load_data(FILE_PATH)
except Exception as e:
    st.error(f"Failed to load data file '{FILE_PATH}': {e}")
    st.stop()

# ==============================
# Sidebar / Inputs
# ==============================
exclude_cols = ['id', 'year', 'state_name', 'state_code', 'district_name', 'district_code', 'registration_circles']

states = sorted(df['state_name'].dropna().unique())
crime_columns = [c for c in df.columns if c not in exclude_cols]

col_left, col_mid, col_right = st.columns([1.2, 1, 1])

with col_left:
    selected_state = st.selectbox("Select a State", states, index=0)

with col_mid:
    selected_crime = st.selectbox("Select Crime Type", crime_columns, index=0)

with col_right:
    forecast_years = st.number_input("Forecast Years", min_value=1, max_value=20, value=FORECAST_YEARS_DEFAULT, step=1)

st.caption("Tip: Use the toggle below to visualize all model forecasts or only the selected best model.")
show_all_models = st.toggle("Show all model lines (in addition to the best model)", value=False)

# ==============================
# Prepare Series
# ==============================
crime_df = df.groupby(['year', 'state_name'])[selected_crime].sum().reset_index()
state_data = crime_df[crime_df['state_name'] == selected_state].sort_values('year')
years_all = state_data['year'].astype(int).values
values_all = state_data[selected_crime].astype(float).values

if len(values_all) < MIN_POINTS_FOR_ARIMA:
    st.warning(f"{selected_state} does not have enough data for forecasting {selected_crime}. Need ≥ {MIN_POINTS_FOR_ARIMA} points.")
    st.stop()

# ==============================
# Train/Validation Split
# ==============================
train_years, train_values, val_years, val_values = split_series(years_all, values_all, TEST_HORIZON)

# ==============================
# Model Wrappers
# ==============================
def arima_fit_forecast(train_vals, val_h, future_h):
    from statsmodels.tsa.arima.model import ARIMA
    # simple order; you can grid search later
    model = ARIMA(train_vals, order=(1,1,1))
    fit = model.fit()
    # validation forecast
    val_fc = fit.forecast(steps=val_h) if val_h > 0 else np.array([])
    # future forecast
    fut_fc_res = fit.get_forecast(steps=future_h)
    fut_fc = fut_fc_res.predicted_mean
    fut_ci = fut_fc_res.conf_int(alpha=0.05)  # DataFrame with lower/upper
    return fit, val_fc, fut_fc, fut_ci.values  # return CI as array (n, 2)

def prophet_fit_forecast(train_yrs, train_vals, val_h, future_h):
    # Prophet requires ds,y
    dfp = pd.DataFrame({"ds": pd.to_datetime(train_yrs, format="%Y"), "y": train_vals})
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    m.fit(dfp)
    total_h = val_h + future_h
    last_year = int(pd.to_datetime(train_yrs[-1], format="%Y").year)
    # build future dataframe with annual frequency
    future_ds = pd.date_range(start=f"{last_year+1}-01-01", periods=total_h, freq="YS")
    future_df = pd.DataFrame({"ds": future_ds})
    forecast = m.predict(future_df)
    yhat = forecast["yhat"].values
    lower = forecast["yhat_lower"].values
    upper = forecast["yhat_upper"].values
    val_fc = yhat[:val_h] if val_h > 0 else np.array([])
    fut_fc = yhat[val_h:]
    fut_ci = np.vstack([lower[val_h:], upper[val_h:]]).T
    return m, val_fc, fut_fc, fut_ci

def lstm_fit_forecast(train_vals, val_h, future_h, window=LSTM_WINDOW, epochs=400):
    # Scale data 0-1
    scaler = MinMaxScaler(feature_range=(0,1))
    train_vals = train_vals.reshape(-1,1)
    scaled = scaler.fit_transform(train_vals)

    # Build sequences
    X, y = [], []
    for i in range(window, len(scaled)):
        X.append(scaled[i-window:i, 0])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)
    if len(X) < 5:
        raise ValueError("Not enough points for LSTM training.")

    X = X.reshape((X.shape[0], X.shape[1], 1))
    model = Sequential([
        LSTM(32, input_shape=(X.shape[1], 1)),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    es = EarlyStopping(monitor="loss", patience=20, restore_best_weights=True)
    model.fit(X, y, epochs=epochs, batch_size=8, verbose=0, callbacks=[es])

    # Helper to roll forward predictions
    def roll_forecast(last_series, steps):
        seq = scaler.transform(last_series.reshape(-1,1))[-window:, 0]
        preds = []
        for _ in range(steps):
            x_in = np.array(seq[-window:]).reshape((1, window, 1))
            yhat = model.predict(x_in, verbose=0)[0,0]
            preds.append(yhat)
            seq = np.append(seq, yhat)
        preds = np.array(preds).reshape(-1,1)
        return scaler.inverse_transform(preds).ravel()

    # Validation & future forecasts
    val_fc = roll_forecast(train_vals.ravel(), val_h) if val_h > 0 else np.array([])
    fut_fc = roll_forecast(np.concatenate([train_vals.ravel(), val_fc]), future_h)

    # Approximate CI using residuals from training reconstruction
    # Build in-sample predictions to estimate residual spread
    ins_pred_scaled = model.predict(X, verbose=0).ravel()
    ins_true_scaled = y
    ins_resid = (ins_true_scaled - ins_pred_scaled)  # on scaled space
    # map residual sigma back to value space approximately
    # derivative of inverse transform ~ scale factor
    sigma_scaled = np.std(ins_resid) if len(ins_resid) > 1 else 0.0
    # approximate sigma in original scale
    # pick scale using mean derivative over range (simplified)
    data_min, data_max = scaler.data_min_[0], scaler.data_max_[0]
    scale = (data_max - data_min)
    sigma = sigma_scaled * scale

    lower = fut_fc - 1.96 * sigma
    upper = fut_fc + 1.96 * sigma
    fut_ci = np.vstack([lower, upper]).T
    return model, val_fc, fut_fc, fut_ci

# ==============================
# Train / Evaluate Each Model
# ==============================
results = []  # list of dicts with model outputs

# ARIMA
try:
    if len(train_values) >= MIN_POINTS_FOR_ARIMA:
        arima_fit, arima_val_fc, arima_fut_fc, arima_fut_ci = arima_fit_forecast(
            train_values, len(val_values), forecast_years
        )
        arima_val_rmse = rmse(val_values, arima_val_fc) if len(val_values) else np.nan
        arima_val_mape = safe_mape(val_values, arima_val_fc) if len(val_values) else np.nan
        results.append({
            "name": "ARIMA(1,1,1)",
            "val_rmse": arima_val_rmse,
            "val_mape": arima_val_mape,
            "future": arima_fut_fc,
            "future_ci": arima_fut_ci,
            "val_fc": arima_val_fc
        })
except Exception as e:
    st.warning(f"ARIMA failed: {e}")

# Prophet
try:
    if PROPHET_AVAILABLE and len(train_values) >= MIN_POINTS_FOR_PROPHET:
        p_fit, p_val_fc, p_fut_fc, p_fut_ci = prophet_fit_forecast(
            train_years, train_values, len(val_values), forecast_years
        )
        p_val_rmse = rmse(val_values, p_val_fc) if len(val_values) else np.nan
        p_val_mape = safe_mape(val_values, p_val_fc) if len(val_values) else np.nan
        results.append({
            "name": "Prophet",
            "val_rmse": p_val_rmse,
            "val_mape": p_val_mape,
            "future": p_fut_fc,
            "future_ci": p_fut_ci,
            "val_fc": p_val_fc
        })
    elif not PROPHET_AVAILABLE:
        st.info("Prophet not installed — skipping (pip install prophet).")
except Exception as e:
    st.warning(f"Prophet failed: {e}")

# LSTM
try:
    if LSTM_AVAILABLE and len(train_values) >= MIN_POINTS_FOR_LSTM:
        l_fit, l_val_fc, l_fut_fc, l_fut_ci = lstm_fit_forecast(
            train_values.copy(), len(val_values), forecast_years, window=LSTM_WINDOW
        )
        l_val_rmse = rmse(val_values, l_val_fc) if len(val_values) else np.nan
        l_val_mape = safe_mape(val_values, l_val_fc) if len(val_values) else np.nan
        results.append({
            "name": "LSTM",
            "val_rmse": l_val_rmse,
            "val_mape": l_val_mape,
            "future": l_fut_fc,
            "future_ci": l_fut_ci,
            "val_fc": l_val_fc
        })
    elif not LSTM_AVAILABLE:
        st.info("TensorFlow/Keras not available — skipping LSTM (pip install tensorflow).")
except Exception as e:
    st.warning(f"LSTM failed: {e}")

if not results:
    st.error("No model produced forecasts. Please check data length or install optional dependencies.")
    st.stop()

# ==============================
# Model Selection
# ==============================
# Choose best by RMSE, then MAPE as tiebreaker (lower is better)
def model_rank_key(r):
    rm = r["val_rmse"]
    mp = r["val_mape"]
    # handle NaNs: put them at the end
    rm = rm if not np.isnan(rm) else np.inf
    mp = mp if not np.isnan(mp) else np.inf
    return (rm, mp)

results_sorted = sorted(results, key=model_rank_key)
best = results_sorted[0]

# ==============================
# Metrics Table
# ==============================
metrics_df = pd.DataFrame([{
    "Model": r["name"],
    "Validation RMSE": r["val_rmse"],
    "Validation MAPE": r["val_mape"]
} for r in results_sorted])

st.subheader("📐 Validation Metrics (last few years as hold-out)")
st.dataframe(metrics_df.style.format({"Validation RMSE": "{:.2f}", "Validation MAPE": "{:.2%}"}), use_container_width=True)
st.success(f"✅ Selected Best Model: **{best['name']}** (lowest RMSE/_MAPE on validation)")

# ==============================
# Build Plot Series
# ==============================
last_year = int(years_all[-1])
future_years = np.arange(last_year + 1, last_year + 1 + forecast_years, dtype=int)

# For visualization of all models (optional)
model_lines = []
for r in results_sorted:
    model_lines.append((r["name"], r["future"], r["future_ci"]))

# ==============================
# Explainability (simple)
# ==============================
# Basic narrative based on slope & last-window change
def simple_summary(y, fut, label):
    if len(y) >= 2:
        recent_change = y[-1] - y[-2]
    else:
        recent_change = 0
    fut_trend = "rising" if fut.mean() > y[-min(3, len(y)) :].mean() else "stable/declining"
    return f"{label}: recent change={recent_change:.1f}; outlook appears **{fut_trend}** over the next {len(fut)} years."

st.subheader("🧠 Explainable Output (Quick Summary)")
for r in results_sorted:
    st.markdown(f"- {simple_summary(values_all, r['future'], r['name'])}")

# ==============================
# Plot
# ==============================
st.subheader(f"📈 {selected_crime.replace('_',' ').title()} Forecast for {selected_state}")

fig, ax = plt.subplots(figsize=(14, 6))

# Actuals
ax.plot(years_all, values_all, marker='o', linestyle='-', label='Actual')
ax.scatter(years_all, values_all)

# Anomaly highlight on historical values
mask_anom = zscore_anomalies(values_all, threshold=ANOMALY_Z)
if mask_anom.any():
    ax.scatter(years_all[mask_anom], values_all[mask_anom], s=80, marker='x', label='Historical Anomaly')

# All models (faint), then best model (bold)
if show_all_models:
    for name, fut, ci in model_lines:
        ax.plot(future_years, fut, linestyle='--', alpha=0.5, label=f"{name} (forecast)")
        if ci is not None and np.array(ci).ndim == 2 and ci.shape[1] == 2:
            ax.fill_between(future_years, ci[:,0], ci[:,1], alpha=0.1)

# Best model emphasized
ax.plot(future_years, best["future"], linestyle='--', linewidth=2.5, label=f"Best: {best['name']} (forecast)")
if best["future_ci"] is not None and np.array(best["future_ci"]).ndim == 2 and best["future_ci"].shape[1] == 2:
    ax.fill_between(future_years, best["future_ci"][:,0], best["future_ci"][:,1], alpha=0.2, label="Confidence Interval")

# Labels & cosmetics
for i, (yr, val) in enumerate(zip(years_all, values_all)):
    ax.text(yr, val, f"{int(val)}", ha='center', va='bottom', fontsize=7)
for i, (yr, val) in enumerate(zip(future_years, best["future"])):
    ax.text(yr, val, f"{int(val)}", ha='center', va='bottom', fontsize=7)

ax.set_title(f"{selected_crime.replace('_',' ').title()} – {selected_state}")
ax.set_xlabel("Year")
ax.set_ylabel(f"Number of {selected_crime.replace('_',' ').title()}")
ax.grid(True)
ax.legend()
st.pyplot(fig)

# ==============================
# Policy Hint (very light PoC)
# ==============================
st.subheader("🏛️ Policy Recommendation (Prototype)")
hist_mean = np.mean(values_all[-min(5, len(values_all)):]) if len(values_all) else 0
fut_mean = np.mean(best["future"])
if fut_mean > 1.2 * hist_mean:
    st.warning("🚨 Rising trend forecasted (>20% above recent average). Consider targeted patrols, hot-spot policing, and community awareness campaigns.")
elif fut_mean < 0.9 * hist_mean:
    st.info("✅ Forecast suggests improvement. Maintain initiatives that correlate with recent declines.")
else:
    st.info("ℹ️ Forecast roughly stable. Continue monitoring and evaluate localized interventions.")

st.caption("Models used: ARIMA, Prophet (if available), LSTM (if TensorFlow available). Selection based on validation RMSE/MAPE. CI native for ARIMA/Prophet; LSTM CI approximated from residuals.")
