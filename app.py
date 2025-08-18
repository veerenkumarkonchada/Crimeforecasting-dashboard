import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Visualization imports with fallbacks
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    import matplotlib.pyplot as plt

# Modeling imports with fallbacks
try:
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

# ========== SETUP ==========
st.set_page_config(
    page_title="Crime Forecasting Dashboard",
    layout="wide",
    page_icon="📊"
)

# ========== DATA LOADING ==========
@st.cache_data
def load_data():
    try:
        # Replace with your actual data loading logic
        df = pd.DataFrame({
            'year': [2010, 2011, 2012, 2013, 2014, 2015],
            'state_name': ['StateA']*6,
            'crime_count': [100, 120, 90, 150, 130, 160]
        })
        return df
    except Exception as e:
        st.error(f"Data loading failed: {str(e)}")
        return pd.DataFrame()

df = load_data()

# ========== UI COMPONENTS ==========
st.title("Crime Pattern Forecasting")
st.markdown("Predict future crime trends using historical data")

# Model selection
available_models = []
if ARIMA_AVAILABLE:
    available_models.append("ARIMA")
if PROPHET_AVAILABLE:
    available_models.append("Prophet")

if not available_models:
    st.error("No forecasting models available. Please install at least one model package.")
    st.stop()

model_choice = st.selectbox("Select Model", available_models)

# ========== FORECASTING ==========
if st.button("Generate Forecast"):
    with st.spinner("Creating forecast..."):
        try:
            # Prepare data
            train_data = df[['year', 'crime_count']].rename(columns={
                'year': 'ds',
                'crime_count': 'y'
            })
            train_data['ds'] = pd.to_datetime(train_data['ds'], format='%Y')
            
            # Modeling
            if model_choice == "ARIMA" and ARIMA_AVAILABLE:
                model = ARIMA(train_data['y'], order=(1,1,1)).fit()
                forecast = model.forecast(steps=3)
                future_dates = pd.date_range(
                    start=train_data['ds'].max(),
                    periods=4,
                    freq='Y'
                )[1:]
            elif model_choice == "Prophet" and PROPHET_AVAILABLE:
                model = Prophet()
                model.fit(train_data)
                future = model.make_future_dataframe(periods=3, freq='Y')
                forecast = model.predict(future)['yhat'][-3:]
                future_dates = future['ds'][-3:]
            
            # Visualization
            if PLOTLY_AVAILABLE:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=train_data['ds'],
                    y=train_data['y'],
                    name='Historical Data',
                    mode='lines+markers'
                ))
                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=forecast,
                    name='Forecast',
                    mode='lines+markers',
                    line=dict(color='red', dash='dash')
                ))
                fig.update_layout(
                    title="Crime Trend Forecast",
                    xaxis_title="Year",
                    yaxis_title="Crime Count"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                plt.figure(figsize=(10,6))
                plt.plot(train_data['ds'], train_data['y'], label='Historical')
                plt.plot(future_dates, forecast, 'r--', label='Forecast')
                plt.title("Crime Trend Forecast")
                plt.xlabel("Year")
                plt.ylabel("Crime Count")
                plt.legend()
                st.pyplot(plt)
                
            st.success("Forecast generated successfully!")
            
        except Exception as e:
            st.error(f"Forecasting failed: {str(e)}")

# ========== SIDEBAR ==========
st.sidebar.header("Settings")
st.sidebar.markdown("Configure your forecasting parameters")
forecast_years = st.sidebar.slider(
    "Years to Forecast",
    min_value=1,
    max_value=5,
    value=3
)

# ========== FOOTER ==========
st.markdown("---")
st.markdown("""
**Note**: This is a simplified demonstration. 
The full implementation would include:
- Multiple crime types
- State-wise analysis
- Advanced anomaly detection
- Explainable AI components
""")
