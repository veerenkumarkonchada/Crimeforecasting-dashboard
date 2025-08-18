import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error

from statsmodels.tsa.arima.model import ARIMA

# Try importing Prophet
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

# Try importing TensorFlow (for LSTM)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


# -------- HYBRID FORECASTING ENGINE --------
def forecast_with_hybrid(data, forecast_years):
    forecasts = {}
    errors = {}

    # Use last 3 years for validation (if enough data)
    train = data[:-3] if len(data) > 6 else data
    test = data[-3:] if len(data) > 6 else None

    # Prophet
    if PROPHET_AVAILABLE:
        try:
            df = pd.DataFrame({"ds": data.index, "y": data.values})
            model = Prophet()
            model.fit(df)
            future = model.make_future_dataframe(periods=forecast_years, freq="Y")
            forecast = model.predict(future)
            yhat = forecast[["ds", "yhat"]].set_index("ds")["yhat"]
            forecasts["Prophet"] = yhat

            if test is not None:
                df_train = pd.DataFrame({"ds": train.index, "y": train.values})
                model_val = Prophet()
                model_val.fit(df_train)
                future_val = model_val.make_future_dataframe(periods=len(test), freq="Y")
                forecast_val = model_val.predict(future_val)
                yhat_val = forecast_val.set_index("ds")["yhat"].iloc[-len(test):]
                errors["Prophet"] = mean_absolute_percentage_error(test.values, yhat_val.values)
        except Exception as e:
            st.warning(f"Prophet failed: {e}")

    # ARIMA
    try:
        model = ARIMA(train, order=(1,1,1))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=forecast_years)
        forecasts["ARIMA"] = pd.Series(
            forecast,
            index=pd.date_range(data.index[-1] + pd.offsets.YearEnd(),
                                periods=forecast_years, freq="Y")
        )

        if test is not None:
            forecast_val = model_fit.forecast(steps=len(test))
            errors["ARIMA"] = mean_absolute_percentage_error(test.values, forecast_val.values)
    except Exception as e:
        st.warning(f"ARIMA failed: {e}")

    # LSTM
    if TF_AVAILABLE:
        try:
            series = train.values.reshape(-1,1)
            X, y = [], []
            for i in range(len(series)-1):
                X.append(series[i])
                y.append(series[i+1])
            X, y = np.array(X), np.array(y)
            X = X.reshape((X.shape[0], 1, X.shape[1]))

            model = Sequential([
                LSTM(50, activation='relu', input_shape=(X.shape[1], X.shape[2])),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')
            model.fit(X, y, epochs=20, verbose=0)

            x_input = series[-1].reshape((1,1,1))
            preds = []
            for _ in range(forecast_years):
                yhat = model.predict(x_input, verbose=0)
                preds.append(yhat[0,0])
                x_input = np.array(yhat).reshape((1,1,1))

            forecasts["LSTM"] = pd.Series(
                preds,
                index=pd.date_range(data.index[-1] + pd.offsets.YearEnd(),
                                    periods=forecast_years, freq="Y")
            )

            if test is not None:
                preds_val = []
                x_input = series[-1].reshape((1,1,1))
                for _ in range(len(test)):
                    yhat = model.predict(x_input, verbose=0)
                    preds_val.append(yhat[0,0])
                    x_input = np.array(yhat).reshape((1,1,1))
                errors["LSTM"] = mean_absolute_percentage_error(test.values, preds_val)
        except Exception as e:
            st.warning(f"LSTM failed: {e}")

    return forecasts, errors


# -------- STREAMLIT UI --------
st.title("📊 Hybrid Crime Forecasting Dashboard")

# Load your dataset
@st.cache_data
def load_data():
    df = pd.read_csv("crime_data.csv")  # <-- replace with your dataset path
    df["year"] = pd.to_datetime(df["year"], format="%Y")
    return df

df = load_data()

state = st.selectbox("Select a State", df["state_name"].unique())
crime_type = st.selectbox("Select Crime Type", [c for c in df.columns if c not in ["id", "year", "state_name", "state_code", "district_name", "district_code"]])
forecast_years = st.number_input("Forecast Years", 1, 10, 5)

filtered = df[df["state_name"] == state][["year", crime_type]].groupby("year").sum()

if st.button("Run Forecast"):
    if len(filtered) < 6:
        st.error("❌ Not enough historical data for forecasting. Try another state/crime.")
    else:
        series = pd.Series(filtered[crime_type].values, index=filtered.index)

        forecasts, errors = forecast_with_hybrid(series, forecast_years)

        if forecasts:
            if errors:
                best_model = min(errors, key=errors.get)
                st.success(f"✅ Best Model Selected: {best_model} (MAPE: {errors[best_model]:.2f})")
                st.line_chart(forecasts[best_model], height=300)
            else:
                st.warning("All models failed validation — showing all forecasts instead.")
                for model_name, series in forecasts.items():
                    st.line_chart(series, height=300)
        else:
            st.error("❌ All models failed. Please check dependencies or data.")
