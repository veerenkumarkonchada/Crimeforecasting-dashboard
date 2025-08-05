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
st.title("📈 Real-Time Crime Forecasting Dashboard")
st.markdown("This dashboard forecasts murder rates for each Indian state using ARIMA time-series modeling.")

# Upload Excel file
FILE_PATH = "AI PROJECT.xlsx"
uploaded_file = None

if os.path.exists(FILE_PATH):
    uploaded_file = FILE_PATH
    
else:
    # Upload Excel file
    uploaded_file = st.file_uploader("Upload your Excel file",type=["xlsx"])
    
if uploaded_file:
    # Load data
    df = pd.read_excel(uploaded_file, header=1)
    
    # Filter necessary columns
    df = df[['year', 'state_name', 'murder']].dropna()

    # Group by year and state
    summary = df.groupby(['year', 'state_name'])['murder'].sum().reset_index()

    # List of all states
    all_states = summary['state_name'].unique()
    selected_state = st.selectbox("Select a State", sorted(all_states))

    # Forecast and plot
    state_data = summary[summary['state_name'] == selected_state].sort_values('year')
    years = state_data['year']
    murders = state_data['murder']

    if len(murders) < 5:
        st.warning(f"{selected_state} does not have enough data for forecasting.")
    else:
        try:
            # ARIMA modeling
            model = ARIMA(murders, order=(1, 1, 1))
            model_fit = model.fit()
            forecast_years = 6
            forecast = model_fit.forecast(steps=forecast_years)

            # Prepare full data
            future_years = list(range(years.max() + 1, years.max() + forecast_years + 1))
            all_years = list(years) + future_years
            all_murders = list(murders) + list(forecast)

            # Plotting
            fig, ax = plt.subplots(figsize=(14, 6))
            ax.plot(all_years, all_murders, marker='o', label='Forecasted', linestyle='--', color='red')
            ax.plot(years, murders, marker='o', label='Actual', linestyle='-', color='blue')
            ax.scatter(years, murders, color='blue')
            ax.scatter(future_years, forecast, color='red')

            # Data labels
            for i, val in enumerate(all_murders):
                ax.text(all_years[i], val + 2, f'{int(val)}', ha='center', fontsize=7)

            ax.set_title(f"Murder Forecast for {selected_state}")
            ax.set_xlabel("Year")
            ax.set_ylabel("Number of Murders")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Forecasting failed for {selected_state}: {e}")
