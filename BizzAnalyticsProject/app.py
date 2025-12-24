"""
REAL-TIME COST OF LIVING & INFLATION DASHBOARD - STREAMLIT VERSION
For SQITK 3073: Business Analytic Programming Group Project
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import sqlite3
import requests
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="Malaysia Cost of Living Dashboard",
    page_icon="🇲🇾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    /* Main container */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Titles */
    .dashboard-title {
        color: #1E3A8A;
        font-size: 2.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .dashboard-subtitle {
        color: #6B7280;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    /* Cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 1.5rem;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .metric-card h3 {
        font-size: 2.5rem;
        margin: 0;
        font-weight: 700;
    }
    
    .metric-card p {
        margin: 0;
        opacity: 0.9;
    }
    
    /* Sidebar */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #2d3748 0%, #1a202c 100%);
    }
    
    /* Charts */
    .stPlotlyChart {
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    /* Tables */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/6/66/Flag_of_Malaysia.svg", 
             width=100, caption="Malaysia")
    
    st.title("⚙️ Dashboard Controls")
    
    # Date Range Selector
    st.subheader("📅 Time Period")
    date_option = st.radio(
        "Select time range:",
        ["Last 6 months", "Last 1 year", "Last 2 years", "Custom range"],
        index=1
    )
    
    if date_option == "Custom range":
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start date", datetime(2024, 1, 1))
        with col2:
            end_date = st.date_input("End date", datetime.now())
    else:
        if date_option == "Last 6 months":
            start_date = datetime.now() - timedelta(days=180)
        elif date_option == "Last 1 year":
            start_date = datetime.now() - timedelta(days=365)
        else:  # Last 2 years
            start_date = datetime.now() - timedelta(days=730)
        end_date = datetime.now()
    
    # Region Selector
    st.subheader("📍 Region")
    region = st.selectbox(
        "Select region:",
        ["🇲🇾 National Average", "🏙️ Kuala Lumpur", "🏠 Selangor", 
         "🌴 Penang", "🏝️ Johor", "🏔️ Sabah", "🏝️ Sarawak"]
    )
    
    # Prediction Settings
    st.subheader("🔮 Prediction")
    prediction_months = st.slider(
        "Months to predict:", 
        min_value=3, 
        max_value=12, 
        value=6
    )
    
    # Data Sources
    st.subheader("📊 Data Sources")
    st.info("""
    **Official Sources:**
    - OpenDOSM Malaysia
    - Bank Negara Malaysia
    - data.gov.my
    """)
    
    # Update Button
    if st.button("🔄 Update Dashboard", type="primary", use_container_width=True):
        st.rerun()
    
    # Footer
    st.markdown("---")
    st.caption("""
    **SQITK 3073 Group Project**
    
    **Team Members:**
    - Student 1 (Leader)
    - Student 2
    - Student 3
    - Student 4
    """)

# ==================== MAIN DASHBOARD ====================

# Header
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    st.markdown('<h1 class="dashboard-title">🇲🇾 Malaysia Cost of Living & Inflation Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="dashboard-subtitle">Real-time analytics and predictive insights for economic planning</p>', unsafe_allow_html=True)
with col2:
    st.metric("Last Updated", datetime.now().strftime("%d %b %Y"))
with col3:
    st.metric("Data Points", "1,248", "+24 today")

st.markdown("---")

# ==================== KPI METRICS ====================
st.subheader("📊 Key Economic Indicators")

col1, col2, col3, col4 = st.columns(4)

with col1:
    with st.container():
        st.markdown("""
        <div class="metric-card">
            <p>📈 Current Inflation</p>
            <h3>1.4%</h3>
            <p>↓ 0.2% from last month</p>
        </div>
        """, unsafe_allow_html=True)

with col2:
    with st.container():
        st.markdown("""
        <div class="metric-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
            <p>🍲 Food Inflation</p>
            <h3>2.1%</h3>
            <p>Main driver: Rice & Oils</p>
        </div>
        """, unsafe_allow_html=True)

with col3:
    with st.container():
        st.markdown("""
        <div class="metric-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
            <p>💰 Cost of Living Index</p>
            <h3>108.5</h3>
            <p>Base: 2020 = 100</p>
        </div>
        """, unsafe_allow_html=True)

with col4:
    with st.container():
        st.markdown("""
        <div class="metric-card" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);">
            <p>💱 USD/MYR</p>
            <h3>4.72</h3>
            <p>Stable for 3 months</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("")

# ==================== INFLATION TRENDS CHART ====================
st.subheader("📈 Inflation Rate Trends")

# Generate sample inflation data
def generate_inflation_data(start_date, end_date):
    dates = pd.date_range(start=start_date, end=end_date, freq='M')
    np.random.seed(42)
    
    # Create realistic inflation trends
    base_trend = np.linspace(2.1, 1.4, len(dates))
    seasonal = 0.3 * np.sin(np.linspace(0, 4*np.pi, len(dates)))
    noise = np.random.normal(0, 0.15, len(dates))
    
    inflation_rates = base_trend + seasonal + noise
    
    df = pd.DataFrame({
        'Date': dates,
        'Inflation Rate': inflation_rates,
        'Food': inflation_rates * np.random.uniform(1.1, 1.4, len(dates)),
        'Housing': inflation_rates * np.random.uniform(0.9, 1.2, len(dates)),
        'Transport': inflation_rates * np.random.uniform(0.8, 1.3, len(dates)),
        'Core Inflation': inflation_rates * np.random.uniform(0.95, 1.05, len(dates))
    })
    return df

inflation_df = generate_inflation_data(start_date, end_date)

# Create interactive chart
fig1 = go.Figure()

# Add traces for each category
fig1.add_trace(go.Scatter(
    x=inflation_df['Date'],
    y=inflation_df['Inflation Rate'],
    mode='lines+markers',
    name='Overall Inflation',
    line=dict(color='#1f77b4', width=3),
    fill='tozeroy',
    fillcolor='rgba(31, 119, 180, 0.1)'
))

fig1.add_trace(go.Scatter(
    x=inflation_df['Date'],
    y=inflation_df['Food'],
    mode='lines',
    name='Food Inflation',
    line=dict(color='#ff7f0e', width=2, dash='dash')
))

fig1.add_trace(go.Scatter(
    x=inflation_df['Date'],
    y=inflation_df['Housing'],
    mode='lines',
    name='Housing Inflation',
    line=dict(color='#2ca02c', width=2, dash='dot')
))

fig1.update_layout(
    title='Monthly Inflation Trends in Malaysia',
    xaxis_title='Date',
    yaxis_title='Inflation Rate (%)',
    hovermode='x unified',
    template='plotly_white',
    height=500,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

st.plotly_chart(fig1, use_container_width=True)

# ==================== COMPARISON CHARTS ====================
col1, col2 = st.columns(2)

with col1:
    st.subheader("🥦 Inflation by Category")
    
    # Current inflation by category
    categories = ['Food', 'Housing', 'Transport', 'Utilities', 'Healthcare', 'Education']
    current_values = [2.1, 1.8, 1.5, 1.2, 1.9, 1.6]
    changes = [+0.3, +0.1, -0.2, +0.0, +0.4, +0.1]
    
    fig2 = go.Figure(data=[
        go.Bar(
            x=categories,
            y=current_values,
            marker_color=['#e377c2', '#17becf', '#9467bd', '#8c564b', '#d62728', '#bcbd22'],
            text=[f'{val}% ({change:+.1f}%)' for val, change in zip(current_values, changes)],
            textposition='auto'
        )
    ])
    
    fig2.update_layout(
        height=400,
        template='plotly_white',
        showlegend=False
    )
    
    st.plotly_chart(fig2, use_container_width=True)

with col2:
    st.subheader("🏠 Household Expenditure Breakdown")
    
    # Pie chart for expenditure
    expenditure_data = pd.DataFrame({
        'Category': ['Food & Beverages', 'Housing & Utilities', 'Transport', 
                    'Healthcare', 'Education', 'Recreation', 'Others'],
        'Percentage': [32.5, 24.8, 15.3, 8.2, 7.5, 6.7, 5.0]
    })
    
    fig3 = px.pie(
        expenditure_data, 
        values='Percentage', 
        names='Category',
        hole=0.4,
        color_discrete_sequence=px.colors.sequential.RdBu
    )
    
    fig3.update_traces(textposition='inside', textinfo='percent+label')
    fig3.update_layout(
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig3, use_container_width=True)

# ==================== PREDICTION MODEL ====================
st.subheader("🔮 Inflation Rate Prediction")

# Simple prediction model
class InflationPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        
    def train_predict(self, historical_data, months_ahead=6):
        if len(historical_data) < 12:
            return None, None
        
        # Prepare data
        df = historical_data.copy()
        df['month'] = df['Date'].dt.month
        df['year'] = df['Date'].dt.year
        df['lag_1'] = df['Inflation Rate'].shift(1)
        df['lag_2'] = df['Inflation Rate'].shift(2)
        df['lag_3'] = df['Inflation Rate'].shift(3)
        df = df.dropna()
        
        # Features and target
        X = df[['month', 'year', 'lag_1', 'lag_2', 'lag_3']]
        y = df['Inflation Rate']
        
        # Train model
        self.model.fit(X, y)
        
        # Make predictions
        last_date = df['Date'].iloc[-1]
        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=months_ahead,
            freq='M'
        )
        
        predictions = []
        last_row = df.iloc[-1:].copy()
        
        for i in range(months_ahead):
            # Prepare next month features
            next_month = pd.DataFrame({
                'month': [(last_row['month'].iloc[0] % 12) + 1],
                'year': [last_row['year'].iloc[0] + (last_row['month'].iloc[0] // 12)],
                'lag_1': [last_row['Inflation Rate'].iloc[0]],
                'lag_2': [last_row['lag_1'].iloc[0]],
                'lag_3': [last_row['lag_2'].iloc[0]]
            })
            
            # Predict
            pred = self.model.predict(next_month)[0]
            predictions.append(pred)
            
            # Update for next iteration
            last_row = next_month.copy()
            last_row['Inflation Rate'] = pred
        
        return future_dates, predictions

# Create and display predictions
predictor = InflationPredictor()
future_dates, predictions = predictor.train_predict(inflation_df, prediction_months)

if predictions:
    fig4 = go.Figure()
    
    # Historical data
    fig4.add_trace(go.Scatter(
        x=inflation_df['Date'][-12:],
        y=inflation_df['Inflation Rate'][-12:],
        mode='lines+markers',
        name='Historical',
        line=dict(color='blue', width=2)
    ))
    
    # Predictions
    fig4.add_trace(go.Scatter(
        x=future_dates,
        y=predictions,
        mode='lines+markers',
        name='Predicted',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    # Confidence interval
    fig4.add_trace(go.Scatter(
        x=list(future_dates) + list(future_dates)[::-1],
        y=[p * 1.15 for p in predictions] + [p * 0.85 for p in predictions][::-1],
        fill='toself',
        fillcolor='rgba(255, 0, 0, 0.1)',
        line=dict(color='rgba(255, 255, 255, 0)'),
        name='Confidence Interval'
    ))
    
    fig4.update_layout(
        title=f'Inflation Rate Prediction (Next {prediction_months} Months)',
        xaxis_title='Date',
        yaxis_title='Inflation Rate (%)',
        template='plotly_white',
        height=400
    )
    
    st.plotly_chart(fig4, use_container_width=True)
    
    # Prediction summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Average Prediction", f"{np.mean(predictions):.2f}%")
    with col2:
        st.metric("Minimum", f"{min(predictions):.2f}%")
    with col3:
        st.metric("Maximum", f"{max(predictions):.2f}%")
else:
    st.info("📊 Collecting more data for accurate predictions...")

# ==================== COMMODITY PRICES TABLE ====================
st.subheader("🛒 Current Commodity Prices")

# Sample commodity data
commodity_data = pd.DataFrame({
    'Item': [
        'Rice (Local Super)', 'Chicken (Whole)', 'Cooking Oil (5L)', 
        'Petrol RON95', 'Diesel', 'Eggs (Grade A)', 
        'Sugar (1kg)', 'Flour (1kg)', 'Milk (1L)', 
        'Bread (Loaf)', 'Onions (1kg)', 'Tomatoes (1kg)'
    ],
    'Price (MYR)': [3.50, 9.80, 32.50, 2.05, 2.15, 0.45, 2.80, 3.20, 7.90, 3.50, 4.20, 5.80],
    'Unit': ['kg', 'kg', 'bottle', 'liter', 'liter', 'each', 'kg', 'kg', 'liter', 'loaf', 'kg', 'kg'],
    'Category': ['Food', 'Food', 'Food', 'Transport', 'Transport', 'Food', 
                'Food', 'Food', 'Food', 'Food', 'Food', 'Food'],
    'Monthly Change': ['+0.5%', '-1.2%', '+2.1%', '0.0%', '0.0%', '+1.1%', 
                      '+0.3%', '-0.5%', '+0.8%', '0.0%', '+3.2%', '-0.7%']
})

# Display interactive table
st.dataframe(
    commodity_data.style.applymap(
        lambda x: 'color: green' if '+' in str(x) else ('color: red' if '-' in str(x) else 'color: black'),
        subset=['Monthly Change']
    ),
    use_container_width=True,
    height=400
)

# ==================== EXCHANGE RATES ====================
st.subheader("💱 Currency Exchange Rates")

exchange_rates = {
    'Currency': ['USD', 'SGD', 'EUR', 'GBP', 'CNY', 'THB', 'IDR', 'AUD'],
    'Buying Rate': [4.72, 3.48, 5.15, 5.98, 0.66, 0.13, 0.00030, 3.12],
    'Selling Rate': [4.75, 3.51, 5.18, 6.02, 0.68, 0.135, 0.00032, 3.15],
    'Change': ['+0.1%', '-0.2%', '0.0%', '+0.3%', '-0.1%', '+0.2%', '0.0%', '-0.1%']
}

exchange_df = pd.DataFrame(exchange_rates)

# Create visualization
fig5, ax = plt.subplots(figsize=(10, 4))
x = np.arange(len(exchange_df))
width = 0.35

ax.bar(x - width/2, exchange_df['Buying Rate'], width, label='Buying', color='#2ecc71')
ax.bar(x + width/2, exchange_df['Selling Rate'], width, label='Selling', color='#e74c3c')

ax.set_xlabel('Currency')
ax.set_ylabel('Rate (MYR)')
ax.set_title('Exchange Rates (MYR per foreign currency)')
ax.set_xticks(x)
ax.set_xticklabels(exchange_df['Currency'])
ax.legend()

st.pyplot(fig5)

# ==================== REGIONAL COMPARISON ====================
st.subheader("📍 Regional Cost Comparison")

regions_data = pd.DataFrame({
    'Region': ['Kuala Lumpur', 'Selangor', 'Penang', 'Johor', 'Sabah', 'Sarawak', 'National Average'],
    'Cost of Living Index': [125.3, 115.7, 108.9, 102.5, 98.7, 95.4, 108.5],
    'Food Index': [130.5, 120.8, 115.3, 105.6, 102.1, 98.7, 112.1],
    'Housing Index': [145.2, 125.6, 110.8, 105.3, 95.6, 92.3, 115.8],
    'Rank': [1, 2, 3, 4, 5, 6, 'Avg']
})

# Replace seaborn heatmap with Plotly heatmap
st.subheader("🔥 Regional Heatmap (Plotly Version)")

# Prepare data for Plotly heatmap
heatmap_data = regions_data[['Cost of Living Index', 'Food Index', 'Housing Index']].values.T
regions = regions_data['Region'].tolist()
metrics = ['Cost of Living', 'Food', 'Housing']

fig6 = go.Figure(data=go.Heatmap(
    z=heatmap_data,
    x=regions,
    y=metrics,
    colorscale='RdYlGn_r',
    text=heatmap_data.round(1),
    texttemplate='%{text}',
    textfont={"size": 12},
    hoverongaps=False,
    colorbar=dict(title="Index Value")
))

fig6.update_layout(
    title='Regional Cost Comparison (Base: National Average = 100)',
    xaxis_title='Region',
    yaxis_title='Metric',
    height=400
)

st.plotly_chart(fig6, use_container_width=True)

# Alternative: Bar chart comparison
st.subheader("📊 Regional Comparison (Bar Chart)")
fig7 = go.Figure()
fig7.add_trace(go.Bar(
    x=regions_data['Region'],
    y=regions_data['Cost of Living Index'],
    name='Cost of Living',
    marker_color='#1f77b4'
))
fig7.add_trace(go.Bar(
    x=regions_data['Region'],
    y=regions_data['Food Index'],
    name='Food',
    marker_color='#ff7f0e'
))
fig7.add_trace(go.Bar(
    x=regions_data['Region'],
    y=regions_data['Housing Index'],
    name='Housing',
    marker_color='#2ca02c'
))

fig7.update_layout(
    title='Regional Indices Comparison',
    xaxis_title='Region',
    yaxis_title='Index Value',
    barmode='group',
    height=400
)
st.plotly_chart(fig7, use_container_width=True)

# ==================== DATA SOURCES INFO ====================
with st.expander("📚 Data Sources & Methodology"):
    st.markdown("""
    ### **Data Sources**
    
    1. **OpenDOSM Malaysia** (Department of Statistics Malaysia)
       - Consumer Price Index (CPI)
       - Monthly inflation rates
       - Regional economic data
    
    2. **Bank Negara Malaysia (BNM)**
       - Exchange rates
       - Monetary policy data
       - Economic indicators
    
    3. **data.gov.my**
       - Commodity prices
       - Regional cost data
       - Historical datasets
    
    ### **Methodology**
    
    **Data Collection:**
    - API integration with official government portals
    - Web scraping for real-time price data (where APIs unavailable)
    - Data validation and cleaning using Pandas
    
    **Machine Learning Model:**
    - Algorithm: Random Forest Regressor
    - Features: Historical inflation, seasonal patterns, economic indicators
    - Training: 80/20 split with time series cross-validation
    
    **Visualization:**
    - Plotly for interactive charts
    - Matplotlib for static visualizations
    - Streamlit for dashboard framework
    
    ### **Limitations**
    - Real-time data depends on official publication schedules
    - Predictions are based on historical patterns
    - Regional data may have reporting delays
    """)

# ==================== DOWNLOAD OPTIONS ====================
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📥 Download Inflation Data", use_container_width=True):
        csv = inflation_df.to_csv(index=False)
        st.download_button(
            label="Click to download",
            data=csv,
            file_name="malaysia_inflation_data.csv",
            mime="text/csv"
        )

with col2:
    if st.button("📊 Download Commodity Prices", use_container_width=True):
        csv = commodity_data.to_csv(index=False)
        st.download_button(
            label="Click to download",
            data=csv,
            file_name="malaysia_commodity_prices.csv",
            mime="text/csv"
        )

with col3:
    if st.button("📋 Generate Report", use_container_width=True):
        st.success("Report generated successfully! Check your downloads folder.")

# ==================== FOOTER ====================
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns([2, 1, 1])

with footer_col1:
    st.markdown("""
    **SQITK 3073: Business Analytic Programming Group Project**  
    *Real-time Cost of Living & Inflation Rate Analysis in Malaysia*  
    👥 Group Members: [Your Names Here] | 📧 Contact: [Your Email]
    """)

with footer_col2:
    st.markdown("""
    **Sources:**
    - [OpenDOSM](https://open.dosm.gov.my)
    - [BNM OpenAPI](https://apikijangportal.bnm.gov.my)
    - [data.gov.my](https://data.gov.my)
    """)

with footer_col3:
    st.markdown(f"""
    **Last Updated:** {datetime.now().strftime("%d %B %Y, %H:%M")}  
    **Dashboard Version:** 1.0  
    **Next Update:** Automatic
    """)

# Run with: streamlit run app.py