import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from streamlit_option_menu import option_menu
import time

# ---------------------------------------------------------
# 1. Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="FraudGuard AI | Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# 2. CSS Styling
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# 3. Load Data & Model
# ---------------------------------------------------------
DATA_PATH = "Final_fraud_dataset.csv"
MODEL_PATH = "FraudAI_model.pkl" 

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
        return model
    except FileNotFoundError:
        st.warning("⚠️ Model file not found. Running in UI Demo Mode.")
        return None

df = load_data()
model = load_model()

# ---------------------------------------------------------
# 4. Sidebar Menu
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("<h1 style='text-align: center; font-size: 60px;'>🛡️</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>FraudGuard AI</h2>", unsafe_allow_html=True)
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

# ---------------------------------------------------------
# 5. Dashboard Section
# ---------------------------------------------------------
if selected == "Dashboard":
    st.title("📊 Historical Data Overview")
    st.markdown("Insights derived from historical transaction records.")
    
    if not df.empty:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Transactions", f"{len(df):,}")
        col2.metric("Avg Amount", f"${df['transaction_amount'].mean():.2f}")
        col3.metric("Merchants", df['merchant_category'].nunique())
        
        target_col = [c for c in df.columns if 'fraud' in c.lower() or 'target' in c.lower() or 'class' in c.lower()]
        if target_col:
            fraud_count = df[target_col[0]].value_counts().get(1, 0)
            col4.metric("Fraud Cases", f"{fraud_count}")
        else:
            col4.metric("Fraud Cases", "N/A")

        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Transactions by Device")
            if 'device_type' in df.columns:
                fig_device = px.pie(df, names='device_type', title="Device Distribution", hole=0.5, color_discrete_sequence=px.colors.sequential.RdBu)
                st.plotly_chart(fig_device, use_container_width=True)
        
        with c2:
            st.subheader("Amount Distribution")
            fig_hist = px.histogram(df, x="transaction_amount", nbins=50, title="Transaction Amounts", color_discrete_sequence=['#007BFF'])
            st.plotly_chart(fig_hist, use_container_width=True)

# ---------------------------------------------------------
# 6. Real-Time Prediction Section (UPDATED)
# ---------------------------------------------------------
elif selected == "Real-Time Prediction":
    st.title("🛡️ Transaction Scanner")
    st.markdown("Enter transaction details below to estimate fraud probability.")

    # --- DEFINING EXPLICIT MAPPINGS ---
    Transaction_Type_dict = {'ATM Withdrawal': 0, 'Bank Transfer': 1, 'Online': 2, 'POS': 3}
    Device_Type_dict = {'Laptop': 0, 'Mobile': 1, 'Tablet': 2}
    Location_dict = {'London': 0, 'Mumbai': 1, 'New York': 2, 'Sydney': 3, 'Tokyo': 4}
    Merchant_Category_dict = {'Clothing': 0, 'Electronics': 1, 'Groceries': 2, 'Restaurants': 3, 'Travel': 4}
    Card_Type_dict = {'Amex': 0, 'Discover': 1, 'Mastercard': 2, 'Visa': 3}
    Authentication_Method_dict = {'Biometric': 0, 'OTP': 1, 'PIN': 2, 'Password': 3}

    with st.container():
        st.subheader("Transaction Details")
        
        c1, c2, c3 = st.columns(3)
        
        with c1:
            transaction_amount = st.number_input("Amount ($)", min_value=0.0, step=10.0, value=100.0)
            
            # Transaction Type
            selected_trans = st.selectbox("Type", list(Transaction_Type_dict.keys()))
            transaction_type_val = Transaction_Type_dict[selected_trans]

            account_balance = st.number_input("Account Balance", min_value=0.0, value=5000.0)
            
            # Device Type
            selected_device = st.selectbox("Device", list(Device_Type_dict.keys()))
            device_type_val = Device_Type_dict[selected_device]
            
            # Location
            selected_loc = st.selectbox("Location", list(Location_dict.keys()))
            location_val = Location_dict[selected_loc]
            
            # Merchant
            selected_merch = st.selectbox("Merchant", list(Merchant_Category_dict.keys()))
            merchant_val = Merchant_Category_dict[selected_merch]

        with c2:
            daily_transaction_count = st.number_input("Daily Count", min_value=0, value=1)
            avg_transaction_amount_7d = st.number_input("Avg Amount (7d)", min_value=0.0, value=50.0)
            failed_transaction_count_7d = st.number_input("Failed Count (7d)", min_value=0)
            
            # Card Type
            selected_card = st.selectbox("Card Type", list(Card_Type_dict.keys()))
            card_type_val = Card_Type_dict[selected_card]
            
            card_age = st.number_input("Card Age (Days)", min_value=0, value=365)
            transaction_distance = st.number_input("Distance (km)", min_value=0.0)

        with c3:
            # Authentication Method
            selected_auth = st.selectbox("Auth Method", list(Authentication_Method_dict.keys()))
            auth_method_val = Authentication_Method_dict[selected_auth]

            is_weekend_val = st.selectbox("Is Weekend?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            
            hour = st.slider("Hour of Day", 0, 23, 12)
            day = st.selectbox("Day", range(1, 32))
            month = st.selectbox("Month", range(1, 13))
            
            days_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
            day_of_week_label = st.selectbox("Weekday", options=list(days_map.values()))
            day_of_week_val = [k for k, v in days_map.items() if v == day_of_week_label][0]

    st.markdown("---")
    center_col1, center_col2, center_col3 = st.columns([1, 2, 1])
    
    with center_col2:
        predict_btn = st.button("🚀 Analyze Transaction", use_container_width=True)

    if predict_btn and model:
        # Construct DataFrame with exactly the same columns used in training
        input_data = pd.DataFrame([[
            transaction_amount, 
            transaction_type_val, 
            account_balance, 
            device_type_val,
            location_val, 
            merchant_val, 
            daily_transaction_count,
            avg_transaction_amount_7d, 
            failed_transaction_count_7d, 
            card_type_val,
            card_age, 
            transaction_distance, 
            auth_method_val, 
            is_weekend_val,
            hour, 
            day, 
            month, 
            day_of_week_val
        ]], columns=[
            "transaction_amount", "transaction_type", "account_balance", "device_type",
            "location", "merchant_category", "daily_transaction_count",
            "avg_transaction_amount_7d", "failed_transaction_count_7d", "card_type",
            "card_age", "transaction_distance", "authentication_method", "is_weekend",
            "hour", "day", "month", "day_of_week"
        ])

        with st.spinner('🔍 AI is scanning patterns...'):
            time.sleep(1) 
            try:
                # Get Probability
                probability = model.predict_proba(input_data)[0][1]
            except Exception as e:
                st.error(f"Prediction Error: {e}")
                probability = 0.0

        # Logic for UI Display
        if probability > 0.5:
            risk_level = "CRITICAL RISK"
            risk_color = "#FF4B4B"
            risk_icon = "🛡️❌"
            risk_message = "Transaction Blocked - High Fraud Probability"
            bar_width = "100%"
        elif probability > 0.3:
            risk_level = "WARNING"
            risk_color = "#FFA500"
            risk_icon = "⚠️"
            risk_message = "Manual Review Required"
            bar_width = "60%"
        else:
            risk_level = "SAFE"
            risk_color = "#00CC96"
            risk_icon = "🛡️✅"
            risk_message = "Transaction Verified Successfully"
            bar_width = "5%"
            
        st.subheader("📋 Security Analysis")
        
        st.markdown(f"""
<style>
.security-card {{
    background-color: white;
    border-radius: 15px;
    padding: 30px;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    border-top: 8px solid {risk_color};
    font-family: sans-serif;
}}
.risk-label {{
    color: #888;
    font-size: 14px;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-weight: bold;
    margin-bottom: 10px;
}}
.main-status {{
    color: {risk_color};
    font-size: 36px;
    font-weight: 900;
    margin: 15px 0;
}}
.icon-display {{
    font-size: 70px;
}}
.risk-bar-bg {{
    background-color: #f0f0f0;
    border-radius: 10px;
    height: 12px;
    width: 100%;
    margin-top: 20px;
    overflow: hidden;
}}
.risk-bar-fill {{
    background-color: {risk_color};
    height: 100%;
    width: {bar_width};
    transition: width 1s ease-in-out;
}}
.labels-row {{
    display: flex; 
    justify-content: space-between; 
    font-size: 12px; 
    color: #999; 
    margin-top: 25px;
}}
</style>

<div class="security-card">
<div class="risk-label">Analysis Result</div>
<div class="icon-display">{risk_icon}</div>
<div class="main-status">{risk_level}</div>
<p style="color: #555; font-size: 18px;">{risk_message}</p>

<div class="labels-row">
<span>Safe</span>
<span>Suspicious</span>
<span>Dangerous</span>
</div>
<div class="risk-bar-bg">
<div class="risk-bar-fill"></div>
</div>
</div>
""", unsafe_allow_html=True)
