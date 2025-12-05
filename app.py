import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import time
import os

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
# 🎨 CUSTOM CSS & STYLING
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
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------
# 📂 DATA & MODEL LOADING
# ------------------------------------------------------
DATA_PATH = "Final_fraud_dataset.csv"
MODEL_PATH = "fraud_model.pkl" # تأكد أن هذا هو اسم الموديل الذي حفظته من النوت بوك

@st.cache_data
def load_data():
    try:
        df = pd.read_csv(DATA_PATH)
        # تطبيق نفس تنظيف الأسماء الذي تم في النوت بوك
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
        
        # تحويل الوقت لاستخراج الخصائص للعرض في الداشبورد
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['day'] = df['timestamp'].dt.day
            df['month'] = df['timestamp'].dt.month
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            
        return df
    except FileNotFoundError:
        st.error("❌ Dataset file not found. Please check the path.")
        return pd.DataFrame()

# دالة إصلاح الموديل (لضمان التوافق مع النسخ المختلفة)
def patch_model_recursive(model):
    try:
        if hasattr(model, "estimators_"):
            for estimator in model.estimators_:
                patch_model_recursive(estimator)
        elif hasattr(model, "steps"):
            for _, step in model.steps:
                patch_model_recursive(step)
        else:
            if not hasattr(model, "monotonic_cst"):
                setattr(model, "monotonic_cst", None)
    except Exception:
        pass

@st.cache_resource
def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        patch_model_recursive(model)
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
    # استخدام أيقونة نصية بدلاً من الصورة لتجنب مشاكل المسارات
    st.markdown("<h1 style='text-align: center; font-size: 60px;'>🛡️</h1>", unsafe_allow_html=True)
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
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Transactions", f"{len(df):,}")
        col2.metric("Avg Amount", f"${df['transaction_amount'].mean():.2f}")
        col3.metric("Merchants", df['merchant_category'].nunique())
        
        if 'fraud_label' in df.columns:
            fraud_count = df['fraud_label'].value_counts().get(1, 0)
            col4.metric("Fraud Cases", f"{fraud_count}")
        else:
            col4.metric("Fraud Cases", "N/A")

        st.markdown("---")
        
        # Charts
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

# ------------------------------------------------------
# 🔍 2. REAL-TIME PREDICTION
# ------------------------------------------------------
elif selected == "Real-Time Prediction":
    st.title("🛡️ Transaction Scanner")
    st.markdown("Enter transaction details below to estimate fraud probability.")

    # دالة مساعدة لتحويل النصوص إلى أرقام (Label Encoding Simulation)
    # هذا ضروري لأن الموديل تدرب على أرقام (0, 1, 2...) وليس نصوص
    def encode_input(column_name, user_selection):
        if column_name in df.columns:
            # الحصول على القيم الفريدة وترتيبها أبجدياً (كما يفعل LabelEncoder)
            unique_values = sorted(df[column_name].unique())
            # إرجاع الـ index المقابل للقيمة المختارة
            try:
                return unique_values.index(user_selection)
            except ValueError:
                return 0
        return 0

    # --- INPUT FORM ---
    with st.container():
        st.subheader("Transaction Details")
        
        # تقسيم المدخلات لثلاثة أعمدة
        c1, c2, c3 = st.columns(3)
        
        with c1:
            # المدخلات الرقمية المباشرة
            transaction_amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=100.0)
            account_balance = st.number_input("Account Balance ($)", min_value=0.0, value=5000.0)
            
            # المدخلات الفئوية (نصوص)
            # نستخدم القيم الأصلية من الداتا فريم قبل التشفير للعرض في القائمة
            tx_types = sorted(df['transaction_type'].unique()) if 'transaction_type' in df.columns else ['POS', 'Online']
            transaction_type_str = st.selectbox("Transaction Type", tx_types)
            
            devices = sorted(df['device_type'].unique()) if 'device_type' in df.columns else ['Mobile', 'Desktop']
            device_type_str = st.selectbox("Device Type", devices)

            locations = sorted(df['location'].unique()) if 'location' in df.columns else ['City A', 'City B']
            location_str = st.selectbox("Location", locations)
            
            merchants = sorted(df['merchant_category'].unique()) if 'merchant_category' in df.columns else ['Retail', 'Travel']
            merchant_category_str = st.selectbox("Merchant Category", merchants)

        with c2:
            daily_transaction_count = st.number_input("Daily Transaction Count", min_value=0, value=1)
            avg_transaction_amount_7d = st.number_input("Avg Amount (Last 7 Days)", min_value=0.0, value=50.0)
            failed_transaction_count_7d = st.number_input("Failed Counts (Last 7 Days)", min_value=0)
            
            card_types = sorted(df['card_type'].unique()) if 'card_type' in df.columns else ['Visa', 'MasterCard']
            card_type_str = st.selectbox("Card Type", card_types)
            
            card_age = st.number_input("Card Age (Days)", min_value=0, value=365)
            transaction_distance = st.number_input("Distance from Home (km)", min_value=0.0, value=10.0)

        with c3:
            auth_methods = sorted(df['authentication_method'].unique()) if 'authentication_method' in df.columns else ['PIN', 'Biometric']
            authentication_method_str = st.selectbox("Auth Method", auth_methods)
            
            is_weekend_val = st.selectbox("Is Weekend?", ["No", "Yes"])
            is_weekend = 1 if is_weekend_val == "Yes" else 0
            
            # ميزات الوقت المستخرجة
            hour = st.slider("Hour of Day", 0, 23, 12)
            day = st.slider("Day of Month", 1, 31, 15)
            month = st.slider("Month", 1, 12, 6)
            
            days_map = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"}
            day_of_week_str = st.selectbox("Day of Week", list(days_map.values()))
            # تحويل اسم اليوم إلى رقم (0-6)
            day_of_week = list(days_map.keys())[list(days_map.values()).index(day_of_week_str)]

    # --- PREDICTION LOGIC ---
    st.markdown("---")
    center_col1, center_col2, center_col3 = st.columns([1, 2, 1])
    
    with center_col2:
        predict_btn = st.button("🚀 Analyze Transaction", use_container_width=True)

    if predict_btn and model:
        # 1. تشفير المدخلات النصية (Encoding) لتناسب الموديل
        transaction_type_enc = encode_input('transaction_type', transaction_type_str)
        device_type_enc = encode_input('device_type', device_type_str)
        location_enc = encode_input('location', location_str)
        merchant_category_enc = encode_input('merchant_category', merchant_category_str)
        card_type_enc = encode_input('card_type', card_type_str)
        authentication_method_enc = encode_input('authentication_method', authentication_method_str)

        # 2. تجهيز البيانات بنفس ترتيب التدريب في النوت بوك
        # الأعمدة المستخدمة في التدريب (X_train columns)
        input_features = [
            transaction_amount,
            transaction_type_enc,
            account_balance,
            device_type_enc,
            location_enc,
            merchant_category_enc,
            daily_transaction_count,
            avg_transaction_amount_7d,
            failed_transaction_count_7d,
            card_type_enc,
            card_age,
            transaction_distance,
            authentication_method_enc,
            is_weekend,
            hour,
            day,
            month,
            day_of_week
        ]
        
        # أسماء الأعمدة لضمان الترتيب الصحيح (اختياري للعرض، مهم للموديل)
        cols = [
            'transaction_amount', 'transaction_type', 'account_balance', 'device_type', 
            'location', 'merchant_category', 'daily_transaction_count', 
            'avg_transaction_amount_7d', 'failed_transaction_count_7d', 'card_type', 
            'card_age', 'transaction_distance', 'authentication_method', 'is_weekend', 
            'hour', 'day', 'month', 'day_of_week'
        ]
        
        input_df = pd.DataFrame([input_features], columns=cols)

        # 3. التوقع
        with st.spinner('🔍 AI is scanning patterns...'):
            time.sleep(1) 
            try:
                probability = model.predict_proba(input_df)[0][1]
            except Exception as e:
                st.error(f"Prediction Error: {e}")
                probability = 0.0

        # --- DISPLAY RESULTS ---
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
