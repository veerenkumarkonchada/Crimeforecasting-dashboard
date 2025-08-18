import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import numpy as np
import warnings
import os

warnings.filterwarnings("ignore")

# ======================
# Page Config
# ======================
st.set_page_config(page_title="Crime Forecasting Dashboard", layout="wide")
st.title("📊 Real-Time Crime Forecasting Dashboard")
st.markdown("Select a state and crime type to forecast for the next 6 years using a hybrid ARIMA + Regression model.")

# ======================
# Load Data Function
# ======================
@st.cache_data
def load_data():
    # If your file is in same folder as app.py
    file_path = "AI PROJECT(2).xlsx"

    # If inside a subfolder "data/", uncomment below:
    # file_path = os.path.join("data", "AI PROJECT(2).xlsx")

    if not os.path.exists(file_path):
        st.error(f"❌ File not found: {file_path}. Please check that it's uploaded with your app.")
        st.stop()

    df = pd.read_excel(file_path, header=0)
    return df

df = load_data()

# ======================
# User Inputs
# ======================
states = sorted(df['state_name'].unique())
selected_state = st.selectbox("Select a State", states)

exclude_cols = ['id', 'year', 'state_name', 'state_code', 'district_name', 'district_code', 'registration_circles']
crime_columns = [col for col in df.columns if col not in exclude_cols]
selected_crime = st.selectbox("Select Crime Type", crime_columns)

# ======================
# Prepare Data
# ======================
crime_df = df.groupby(['year', 'state_name'])[selected_crime].sum().reset_index()
state_data = crime_df[crime_df['state_name'] == selected_state].sort_values('year')
years = state_data['year'].values.reshape(-1, 1)
values = state_data[selected_crime].values

# ======================
# Hybrid Forecasting
# ======================
if len(values) < 5:
    st.warning(f"{selected_state} does not have enough data for forecasting {selected_crime}.")
else:
    try:
        # ----- ARIMA -----
        arima_model = ARIMA(values, order=(1, 1, 1))
        arima_fit = arima_model.fit()
        arima_forecast = arima_fit.forecast(steps=6)

        # ----- Regression -----
        reg_model = LinearRegression()
        reg_model.fit(years, values)
        future_years = np.arange(years.max() + 1, years.max() + 7).reshape(-1, 1)
        reg_forecast = reg_model.predict(future_years)

        # ----- Hybrid (average) -----
        hybrid_forecast = (arima_forecast + reg_forecast) / 2

        # Combine
        all_years = list(years.flatten()) + list(future_years.flatten())
        all_values = list(values) + list(hybrid_forecast)

        # ======================
        # Plot
        # ======================
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(years.flatten(), values, marker='o', linestyle='-', color='blue', label='Actual')
        ax.plot(future_years.flatten(), hybrid_forecast, marker='o', linestyle='--', color='red', label='Hybrid Forecast')

        ax.set_title(f"{selected_crime} Forecast for {selected_state}")
        ax.set_xlabel("Year")
        ax.set_ylabel(f"Number of {selected_crime.replace('_',' ').title()}")
        ax.legend()
        ax.grid(True)

        st.pyplot(fig)

        # ======================
        # Show Forecast Table
        # ======================
        forecast_df = pd.DataFrame({
            "Year": future_years.flatten(),
            f"Forecasted {selected_crime}": hybrid_forecast.astype(int)
        })
        st.subheader("📌 Forecasted Data")
        st.dataframe(forecast_df)

    except Exception as e:
        st.error(f"Forecasting failed for {selected_state} 
