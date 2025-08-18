import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Visualization import with fallback
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    import matplotlib.pyplot as plt
    st.warning("Plotly not found - using matplotlib instead. For better visuals, run: pip install plotly")

# Sample data loading
@st.cache_data
def load_data():
    return pd.DataFrame({
        'year': [2015, 2016, 2017, 2018, 2019, 2020],
        'crime_count': [100, 120, 90, 150, 130, 160]
    })

# App UI
st.title("Crime Forecasting Dashboard")
df = load_data()

if PLOTLY_AVAILABLE:
    fig = px.line(df, x='year', y='crime_count', title='Crime Trends')
    st.plotly_chart(fig)
else:
    plt.figure(figsize=(10,5))
    plt.plot(df['year'], df['crime_count'])
    plt.title("Crime Trends (matplotlib)")
    plt.xlabel("Year")
    plt.ylabel("Crime Count")
    st.pyplot(plt)

st.write("Sample forecast coming soon...")
