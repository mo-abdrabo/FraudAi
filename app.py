import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import time

# ------------------------------------------------------
# 🔧 PAGE CONFIGURATION
# ------------------------------------------------------
st.set_page_config(
    page_title="FraudGuard AI | Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------
# 🎨 CUSTOM CSS & STYLING (Professional Theme)
# ------------------------------------------------------
st.markdown("""
    <style>
        .block-container {padding-top: 1rem; padding-bottom: 5rem;}
        h1, h2, h3 {font-family: 'Helvetica Neue', sans-serif; color: #0E1117;}
        .stButton>button {
            width: 100%;
            border-radius: 8px;
            height: 3em;
            background-color: #007BFF; 
            color: white;
            font-weight: bold;
            border: none;
        }
        .stButton>button:hover {
            background-color: #0056b3;
        }
        .metric-card {
            background-color: #F0F2F6;
            border-left: 5px solid #007BFF;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------
# 📂 DATA & MODEL LOADING (Cached for Speed)
# ------------------------------------------------------
# ⚠️ UPDATE THESE PATHS TO YOUR LOCAL FILE LOCATIONS
DATA_PATH = "Final_fraud_dataset.csv"
MODEL_PATH = "fraud_model.pkl"


@st.cache_data
def load_data():
    try:
        df = pd.read_csv(DATA_PATH)
        return df
    except FileNotFoundError:
        st.error("❌ Dataset file not found. Please check the path.")
        return pd.DataFrame()

@st.cache_resource
def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        
       
        try:
            if hasattr(model, "estimators_"):
                for estimator in model.estimators_:
                    if not hasattr(estimator, "monotonic_cst"):
                        estimator.monotonic_cst = None
            
            if not hasattr(model, "monotonic_cst"):
                setattr(model, "monotonic_cst", None)
                
        except Exception as e:
            print(f"Patching skipped: {e}")

        return model
    except FileNotFoundError:
        st.warning("⚠️ Model file not found. Running in UI Demo Mode.")
        return None

df = load_data()
model = load_model()

# ------------------------------------------------------
# 🛡️ SIDEBAR & NAVIGATION
# ------------------------------------------------------
with st.sidebar:
    # --- LOGO AREA ---
    # Replace URL with a local image path if needed
    st.image("https://cdn-icons-png.flaticon.com/512/2040/2040520.png", width=90) 
    st.markdown("## **FraudGuard AI**")
    st.caption("Advanced Security System")
    st.markdown("---")
    
    selected = option_menu(
        menu_title="Main Menu",
        options=["Dashboard", "Real-Time Prediction"],
        icons=["bar-chart-fill", "shield-check"],
        menu_icon="cast",
        default_index=1,
        styles={
            "container": {"padding": "0!important", "background-color": "#fafafa"},
            "icon": {"color": "#007BFF", "font-size": "18px"}, 
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px"},
            "nav-link-selected": {"background-color": "#0E1117"},
        }
    )
    st.markdown("---")
    st.info("System Status: **Online** 🟢")

# ------------------------------------------------------
# 🏠 1. DASHBOARD VIEW
# ------------------------------------------------------
if selected == "Dashboard":
    st.title("📊 Historical Data Overview")
    st.markdown("Insights derived from historical transaction records.")
    
    if not df.empty:
        # Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Transactions", f"{len(df):,}")
        col2.metric("Avg Amount", f"${df['transaction_amount'].mean():.2f}")
        col3.metric("Merchants", df['merchant_category'].nunique())
        
        # Identify fraud column flexibly
        target_col = [c for c in df.columns if 'fraud' in c.lower() or 'target' in c.lower() or 'class' in c.lower()]
        if target_col:
            fraud_count = df[target_col[0]].value_counts().get(1, 0)
            col4.metric("Fraud Cases", f"{fraud_count}")
        else:
            col4.metric("Fraud Cases", "N/A")

        # Charts Row
        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Transactions by Device")
            fig_device = px.pie(df, names='device_type', title="Device Distribution", hole=0.5, color_discrete_sequence=px.colors.sequential.RdBu)
            st.plotly_chart(fig_device, use_container_width=True)
        
        with c2:
            st.subheader("Amount Distribution")
            fig_hist = px.histogram(df, x="transaction_amount", nbins=50, title="Transaction Amounts", color_discrete_sequence=['#007BFF'])
            st.plotly_chart(fig_hist, use_container_width=True)

# ------------------------------------------------------
# 🔍 2. REAL-TIME PREDICTION
# ------------------------------------------------------
elif selected == "Real-Time Prediction":
    st.title("🛡️ Transaction Scanner")
    st.markdown("Enter transaction details below to estimate fraud probability.")

    # Helper to find cols regardless of case
    def get_col(name):
        for col in df.columns:
            if name.lower() in col.lower(): return col
        return name

    # --- INPUT FORM LAYOUT ---
    with st.container():
        st.subheader("Transaction Details")
        
        c1, c2, c3 = st.columns(3)
        
        with c1:
            transaction_amount = st.number_input(" Amount ($)", min_value=0.0, step=10.0, value=100.0)
            transaction_type = st.selectbox("Type", df[get_col("transaction_type")].unique())
            account_balance = st.number_input(" Account Balance", min_value=0.0, value=5000.0)
            device_type = st.selectbox(" Device", df[get_col("device_type")].unique())
            location = st.selectbox(" Location", df[get_col("location")].unique())
            merchant_category = st.selectbox(" Merchant", df[get_col("merchant_category")].unique())

        with c2:
            daily_transaction_count = st.number_input(" Daily Count", min_value=0, value=1)
            avg_transaction_amount_7d = st.number_input(" Avg Amount (7d)", min_value=0.0, value=50.0)
            failed_transaction_count_7d = st.number_input(" Failed Count (7d)", min_value=0)
            card_type = st.selectbox(" Card Type", df[get_col("card_type")].unique())
            card_age = st.number_input(" Card Age (Days)", min_value=0, value=365)
            transaction_distance = st.number_input(" Distance (km)", min_value=0.0)

        with c3:
            authentication_method = st.selectbox("Auth Method", df[get_col("authentication_method")].unique())
            is_weekend = st.selectbox(" Is Weekend?", df[get_col("is_weekend")].unique())
            hour = st.slider("Hour of Day", 0, 23, 12)
            day = st.selectbox("Day", df[get_col("day")].unique())
            month = st.selectbox("Month", df[get_col("month")].unique())
            day_of_week = st.selectbox("Weekday", df[get_col("day_of_week")].unique())

    # --- PREDICTION LOGIC ---
    st.markdown("---")
    center_col1, center_col2, center_col3 = st.columns([1, 2, 1])
    
    with center_col2:
        predict_btn = st.button("🚀 Analyze Transaction", use_container_width=True)

    if predict_btn and model:
        # Prepare Data Structure
        required_cols = [
            "transaction_amount", "transaction_type", "account_balance", "device_type",
            "location", "merchant_category", "daily_transaction_count",
            "avg_transaction_amount_7d", "failed_transaction_count_7d", "card_type",
            "card_age", "transaction_distance", "authentication_method", "is_weekend",
            "hour", "day", "month", "day_of_week"
        ]
        
        input_data = pd.DataFrame([[
            transaction_amount, transaction_type, account_balance, device_type,
            location, merchant_category, daily_transaction_count,
            avg_transaction_amount_7d, failed_transaction_count_7d, card_type,
            card_age, transaction_distance, authentication_method, is_weekend,
            hour, day, month, day_of_week
        ]], columns=required_cols)

        # Loading Animation
        with st.spinner('🔍 AI is scanning patterns...'):
            time.sleep(1) # Simulated delay for UX
            try:
                prediction = model.predict(input_data)[0]
                probability = model.predict_proba(input_data)[0][1] # Probability of Fraud
            except Exception as e:
                st.error(f"Prediction Error: {e}")
                prediction = 0
                probability = 0.0

        # --- RESULTS DISPLAY ---
        st.subheader("📋 Analysis Results")
        
        col_res1, col_res2 = st.columns([1, 1])
        
        with col_res1:
            # Professional Gauge Chart
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = probability * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Fraud Risk Score"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#FF4B4B" if probability > 0.5 else "#00CC96"},
                    'steps': [
                        {'range': [0, 40], 'color': "#f8f9fa"},
                        {'range': [40, 70], 'color': "#e9ecef"},
                        {'range': [70, 100], 'color': "#dee2e6"}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90}}))
            st.plotly_chart(fig, use_container_width=True)

        with col_res2:
            st.markdown("### Decision Recommendation:")
            if prediction == 1:
                st.error(f"🚨 **FRAUD DETECTED**\n\nHigh risk transaction detected. Immediate verification recommended.")
            else:
                st.success(f"✅ **LEGITIMATE**\n\nTransaction appears safe to proceed.")
            

            st.info(f"**Model Confidence:** {probability*100:.2f}%")
