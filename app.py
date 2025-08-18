import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.ensemble import IsolationForest
import shap
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ====================== SETUP ======================
st.set_page_config(
    page_title="Hybrid Crime Forecasting Dashboard",
    layout="wide",
    page_icon="🛡️"
)

# ====================== CORE FUNCTIONS ======================
@st.cache_data
def load_data():
    """Load and preprocess crime data"""
    # In production: Connect to federated data sources
    df = pd.read_excel("crime_data.xlsx")  # Replace with your actual data
    df = df.drop(columns=['id', 'registration_circles'], errors='ignore')  # Anonymization
    return df

@st.cache_resource
def train_arima(data):
    """Train ARIMA model"""
    model = ARIMA(data['y'], order=(1,1,1)).fit()
    return model

@st.cache_resource
def train_prophet(data):
    """Train Prophet model"""
    model = Prophet()
    model.fit(data)
    return model

def detect_anomalies(data):
    """Identify statistical anomalies"""
    clf = IsolationForest(contamination=0.1)
    anomalies = clf.fit_predict(data['y'].values.reshape(-1, 1))
    return data[anomalies == -1]

def explain_model(model, model_type, data):
    """Generate SHAP explanations"""
    explainer = shap.Explainer(model.predict, data)
    shap_values = explainer(data['y'])
    return shap_values

# ====================== DASHBOARD LAYOUT ======================
def main():
    # Role-based access control
    st.sidebar.header("Access Control")
    role = st.sidebar.selectbox("Select Role", ["Public", "Law Enforcement", "Policy Maker"])
    
    # Data selection
    df = load_data()
    states = sorted(df['state_name'].unique())
    selected_state = st.sidebar.selectbox("State", states)
    crime_types = [c for c in df.columns if c not in ['year', 'state_name', 'state_code']]
    selected_crime = st.sidebar.selectbox("Crime Type", crime_types)
    
    # Model selection
    model_type = st.sidebar.radio("Model", ["ARIMA", "Prophet"])
    
    # Filter data
    state_data = df[df['state_name'] == selected_state]
    ts_data = state_data.groupby('year')[selected_crime].sum().reset_index()
    ts_data = ts_data.rename(columns={'year': 'ds', selected_crime: 'y'})
    ts_data['ds'] = pd.to_datetime(ts_data['ds'], format='%Y')

    # ================ ANALYSIS SECTION ================
    if st.sidebar.button("Run Full Analysis"):
        with st.spinner("Crunching crime data..."):
            # Train model
            if model_type == "ARIMA":
                model = train_arima(ts_data)
                forecast_steps = st.sidebar.slider("Forecast Years", 1, 5, 3)
                forecast = model.forecast(steps=forecast_steps)
                future_dates = pd.date_range(
                    start=ts_data['ds'].max(),
                    periods=forecast_steps+1,
                    freq='Y'
                )[1:]
            else:  # Prophet
                model = train_prophet(ts_data)
                future = model.make_future_dataframe(periods=3, freq='Y')
                forecast = model.predict(future)['yhat'][-3:]
                future_dates = future['ds'][-3:]

            # ================ VISUALIZATION ================
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(f"{selected_crime.replace('_', ' ').title()} in {selected_state}")
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=ts_data['ds'], y=ts_data['y'],
                    name='Historical Data',
                    mode='lines+markers'
                ))
                fig.add_trace(go.Scatter(
                    x=future_dates, y=forecast,
                    name='Forecast',
                    mode='lines+markers',
                    line=dict(color='red', dash='dash')
                ))
                fig.update_layout(
                    xaxis_title="Year",
                    yaxis_title="Cases Reported",
                    hovermode="x unified"
                )
                st.plotly_chart(fig, use_container_width=True)

            # ================ ANOMALY DETECTION ================
            with col2:
                st.subheader("Anomaly Detection")
                anomalies = detect_anomalies(ts_data)
                if not anomalies.empty:
                    fig_anom = px.scatter(
                        ts_data, x='ds', y='y',
                        title="Suspected Anomalies",
                        color=ts_data.index.isin(anomalies.index),
                        color_discrete_map={True: 'red', False: 'blue'}
                    )
                    st.plotly_chart(fig_anom, use_container_width=True)
                else:
                    st.info("No significant anomalies detected")

            # ================ EXPLAINABILITY ================
            st.subheader("Model Explainability")
            if model_type == "ARIMA":
                shap_values = explain_model(model, model_type, ts_data)
                fig_shap = go.Figure()
                fig_shap.add_trace(go.Bar(
                    x=shap_values.values.flatten(),
                    y=ts_data['ds'].dt.year.astype(str),
                    orientation='h'
                ))
                st.plotly_chart(fig_shap, use_container_width=True)
            
            # ================ POLICY RECOMMENDATIONS ================
            if role != "Public":
                st.subheader("Policy Recommendations")
                last_value = ts_data['y'].iloc[-1]
                forecast_change = ((forecast[0] - last_value)/last_value)*100
                
                if forecast_change > 20:
                    st.warning("🚨 Significant Increase Predicted")
                    st.write(f"Recommended actions for {selected_state}:")
                    st.markdown("""
                    - Increase patrols in high-risk areas
                    - Launch public awareness campaign
                    - Allocate additional resources to crime prevention
                    """)
                else:
                    st.success("Stable trends predicted")

if __name__ == "__main__":
    main()
