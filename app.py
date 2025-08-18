import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import warnings
import os

# Suppress warnings
warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(page_title="Crime Forecasting Dashboard", layout="wide")
st.title("📊 Real-Time Crime Forecasting Dashboard")
st.markdown("Select a state and crime type to forecast for the next 6 years using ARIMA.")

# ===== Load data from file included in the repo =====
FILE_PATH = "AI PROJECT(2).xlsx"
df = pd.read_excel(FILE_PATH, header=0)  # first row as header

# ===== User Inputs =====
states = sorted(df['state_name'].unique())
selected_state = st.selectbox("Select a State", states)

# Detect crime columns (exclude id, year, state info, district info, etc.)
exclude_cols = ['id', 'year', 'state_name', 'state_code', 'district_name', 'district_code', 'registration_circles']
crime_columns = [col for col in df.columns if col not in exclude_cols]
selected_crime = st.selectbox("Select Crime Type", crime_columns)

# ===== Prepare Data =====
crime_df = df.groupby(['year', 'state_name'])[selected_crime].sum().reset_index()
state_data = crime_df[crime_df['state_name'] == selected_state].sort_values('year')
years = state_data['year']
values = state_data[selected_crime]

# ===== Forecast =====
if len(values) < 5:
    st.warning(f"{selected_state} does not have enough data for forecasting {selected_crime}.")
else:
    try:
        model = ARIMA(values, order=(1, 1, 1))
        model_fit = model.fit()

        forecast_years = 6
        forecast = model_fit.forecast(steps=forecast_years)

        future_years = list(range(years.max() + 1, years.max() + forecast_years + 1))
        all_years = list(years) + future_years
        all_values = list(values) + list(forecast)

        # ===== Plot =====
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(all_years, all_values, marker='o', linestyle='--', color='red', label='Forecasted')
        ax.plot(years, values, marker='o', linestyle='-', color='blue', label='Actual')
        ax.scatter(years, values, color='blue')
        ax.scatter(future_years, forecast, color='red')

        for i, val in enumerate(all_values):
            ax.text(all_years[i], val + 2, f'{int(val)}', ha='center', fontsize=7)

        ax.set_title(f"{selected_crime} Forecast for {selected_state}")
        ax.set_xlabel("Year")
        ax.set_ylabel(f"Number of {selected_crime.replace('_',' ').title()}")
        ax.legend()
        ax.grid(True)

        st.pyplot(fig)

    except Exception as e:
        st.error(f"Forecasting failed for {selected_state} - {selected_crime}: {e}")
        
