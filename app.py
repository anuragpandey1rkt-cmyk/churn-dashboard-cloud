import streamlit as st
import pandas as pd
import psycopg2
import plotly.express as px
import plotly.graph_objects as go
from groq import Groq
import io
import hashlib
import time
import datetime
import xlsxwriter

# ==========================================
# 1. CONFIGURATION & PROFESSIONAL UI
# ==========================================
st.set_page_config(
    page_title="RetainIQ: Customer Lifecycle & Churn",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "SaaS Product" Feel
st.markdown("""
    <style>
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e6e6e6;
        border-radius: 8px;
        padding: 20px 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .stMetric label { font-weight: 600; color: #555; }
    .stMetric [data-testid="stMetricValue"] { font-size: 2rem; color: #1E3A8A; }
    h1, h2, h3 { color: #1E3A8A; }
    </style>
""", unsafe_allow_html=True)

# Load Secrets
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", None)
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# ==========================================
# 2. SESSION STATE
# ==========================================
if 'user' not in st.session_state:
    st.session_state.update({
        "user": None, "logged_in": False, "feature": "📊 Dashboard", "churn_df": None, "raw_df": None
    })

def go_to(page): st.session_state.feature = page

# ==========================================
# 3. BACKEND: DATABASE & AUTH
# ==========================================
def get_db_connection():
    try:
        return psycopg2.connect(
            host=st.secrets["DB_HOST"], database=st.secrets["DB_NAME"],
            user=st.secrets["DB_USER"], password=st.secrets["DB_PASS"]
        )
    except: return None

def init_auth_db():
    conn = get_db_connection()
    if conn:
        try:
            conn.cursor().execute("""
                CREATE TABLE IF NOT EXISTS users (
                    email VARCHAR(255) PRIMARY KEY, password_hash VARCHAR(255), name VARCHAR(100)
                );
            """)
            conn.commit()
            conn.close()
        except: pass

init_auth_db()

def hash_password(password): return hashlib.sha256(str.encode(password)).hexdigest()

def login_user(email, password):
    conn = get_db_connection()
    if not conn: st.error("DB Connection Failed"); return
    try:
        cur = conn.cursor()
        cur.execute("SELECT password_hash, name FROM users WHERE email = %s", (email,))
        data = cur.fetchone()
        if data and data[0] == hash_password(password):
            st.session_state.user, st.session_state.logged_in = email, True
            st.success(f"Welcome, {data[1]}"); time.sleep(0.5); st.rerun()
        else: st.error("Invalid credentials")
    except: st.error("Login Error")
    finally: conn.close()

def signup_user(email, password, name):
    conn = get_db_connection()
    if not conn: st.error("DB Connection Failed"); return
    try:
        conn.cursor().execute("INSERT INTO users (email, password_hash, name) VALUES (%s, %s, %s)",
                              (email, hash_password(password), name))
        conn.commit()
        st.success("Account created! Login now.")
    except: st.error("User exists")
    finally: conn.close()

def logout_user():
    st.session_state.clear()
    st.rerun()

# ==========================================
# 4. ADVANCED CHURN ENGINE (The Brain)
# ==========================================

def process_advanced_analytics(df):
    """
    Derives Churn Risk, Lifecycle Stage, and Scores from raw transactions.
    """
    # 1. Standardization
    df.columns = df.columns.str.strip()
    col_map = {}
    for c in df.columns:
        if 'Date' in c: col_map[c] = 'Date'
        elif 'Name' in c or 'Customer' in c: col_map[c] = 'Customer'
        elif 'Amount' in c or 'Sales' in c: col_map[c] = 'Amount'
        elif 'Product' in c: col_map[c] = 'Item'
    df = df.rename(columns=col_map)
    
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(0)
    df['Item'] = df['Item'].fillna("General")
    
    # Snapshot Date (Day after last transaction)
    snapshot_date = df['Date'].max() + datetime.timedelta(days=1)

    # 2. Customer Level Aggregation (RFM)
    metrics = df.groupby('Customer').agg({
        'Date': [
            lambda x: (snapshot_date - x.max()).days, # Recency
            'min'                                     # First Purchase Date
        ],
        'Customer': 'count',                          # Frequency
        'Amount': 'sum',                              # Monetary
        'Item': lambda x: x.mode()[0] if not x.mode().empty else "Mix"
    })
    
    # Flatten columns
    metrics.columns = ['Recency', 'First_Purchase_Date', 'Frequency', 'LTV', 'Fav_Product']
    metrics['Days_Since_First_Buy'] = (snapshot_date - metrics['First_Purchase_Date']).dt.days
    
    # 3. SCORING ENGINE (0-100 Risk Score)
    # Higher Score = Higher Risk of Churning
    # Formula: Heavily weighted on Recency (70%) + Frequency (30%)
    
    # Normalize Recency (Cap at 120 days for scoring)
    metrics['Recency_Score'] = metrics['Recency'].apply(lambda x: min(x, 120) / 120 * 100)
    
    # Normalize Frequency (Inverse: Low freq = High risk)
    metrics['Freq_Score'] = metrics['Frequency'].apply(lambda x: 100 if x==1 else max(0, 100 - (x * 5)))
    
    # Composite Risk Score
    metrics['Churn_Risk_Score'] = (0.7 * metrics['Recency_Score']) + (0.3 * metrics['Freq_Score'])
    
    # 4. LIFECYCLE SEGMENTATION (Business Rules)
    def define_lifecycle(row):
        recency = row['Recency']
        freq = row['Frequency']
        first_buy_age = row['Days_Since_First_Buy']
        
        if recency > 90: return "🔴 Churned"
        if recency > 45: return "⚠️ At Risk"
        if first_buy_age < 30: return "✨ New Customer" # Recent acquisition
        if freq >= 5: return "⭐ Loyal"
        return "🟢 Active"

    metrics['Lifecycle_Stage'] = metrics.apply(define_lifecycle, axis=1)
    
    return metrics.reset_index(), df, snapshot_date

# ==========================================
# 5. UI COMPONENTS
# ==========================================

def render_login():
    st.markdown("<div style='text-align: center; padding: 40px;'><h1>🛡️ RetainIQ</h1><p>Enterprise Churn Analytics Suite</p></div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        tab1, tab2 = st.tabs(["Login", "Register"])
        with tab1:
            e = st.text_input("Email", key="l_e")
            p = st.text_input("Password", type="password", key="l_p")
            if st.button("Access Dashboard", use_container_width=True): login_user(e, p)
        with tab2:
            e2 = st.text_input("Email", key="s_e")
            n2 = st.text_input("Name", key="s_n")
            p2 = st.text_input("Password", type="password", key="s_p")
            if st.button("Create Account", use_container_width=True): signup_user(e2, p2, n2)

def render_dashboard():
    st.title("📊 Executive Retention Dashboard")
    st.markdown("Monitor customer health, identify churn risks, and simulate revenue recovery.")
    
    # FILE UPLOAD
    uploaded_file = st.file_uploader("Upload Transaction Data (CSV/Excel)", type=['csv', 'xlsx'])
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'): raw = pd.read_csv(uploaded_file)
            else: raw = pd.read_excel(uploaded_file)
            
            # PROCESS
            churn_df, raw_clean, snap_date = process_advanced_analytics(raw)
            st.session_state.churn_df = churn_df
            st.session_state.raw_df = raw_clean
            
        except Exception as e: st.error(f"Data Error: {e}")

    if st.session_state.churn_df is None:
        st.info("👋 Upload data to begin analysis.")
        return

    df = st.session_state.churn_df
    raw_df = st.session_state.raw_df
    
    # --- 1. KPI ROW ---
    st.markdown("### 📈 Health Overview")
    k1, k2, k3, k4 = st.columns(4)
    
    total_cust = len(df)
    churn_cnt = len(df[df['Lifecycle_Stage'] == "🔴 Churned"])
    risk_cnt = len(df[df['Lifecycle_Stage'] == "⚠️ At Risk"])
    rev_risk = df[df['Lifecycle_Stage'].isin(["🔴 Churned", "⚠️ At Risk"])]['LTV'].sum()
    churn_rate = (churn_cnt / total_cust) * 100
    
    k1.metric("Total Customers", total_cust)
    k2.metric("Churn Rate", f"{churn_rate:.1f}%", "-Goal: <15%", delta_color="inverse")
    k3.metric("High Risk Users", risk_cnt, "Action Needed", delta_color="off")
    k4.metric("Revenue at Risk", f"₹{rev_risk:,.0f}", "Potential Loss")
    
    st.markdown("---")

    # --- 2. TREND & LIFECYCLE ---
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.subheader("🗓️ Monthly Active Users (Retention Trend)")
        st.caption("Tracking unique customers purchasing per month. A downward trend indicates systemic churn.")
        
        # Time Series Logic
        raw_df['Month'] = raw_df['Date'].dt.to_period('M').astype(str)
        monthly_active = raw_df.groupby('Month')['Customer'].nunique().reset_index()
        
        fig_trend = px.area(monthly_active, x='Month', y='Customer', markers=True, 
                            line_shape='spline', color_discrete_sequence=['#4B4BFF'])
        fig_trend.update_layout(xaxis_title=None, yaxis_title="Active Customers")
        st.plotly_chart(fig_trend, use_container_width=True)
        
    with c2:
        st.subheader("👥 Lifecycle Segmentation")
        st.caption("Current distribution of your customer base.")
        
        fig_pie = px.pie(df, names='Lifecycle_Stage', hole=0.5, 
                         color='Lifecycle_Stage',
                         color_discrete_map={
                             "✨ New Customer": "#3498db", "⭐ Loyal": "#8e44ad", 
                             "🟢 Active": "#2ecc71", "⚠️ At Risk": "#f39c12", 
                             "🔴 Churned": "#e74c3c"
                         })
        fig_pie.update_layout(showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)

    # --- 3. RISK SIMULATOR & MATRIX ---
    st.markdown("---")
    r1, r2 = st.columns([1, 2])
    
    with r1:
        st.subheader("💰 Revenue Recovery Simulator")
        st.info("If we launch a campaign today, how much revenue can we save?")
        
        save_rate = st.slider("Target Retention Rate (%)", 0, 100, 20)
        
        # Calculate Potential Savings
        risk_pool = df[df['Lifecycle_Stage'] == "⚠️ At Risk"]['LTV'].sum()
        saved_amount = risk_pool * (save_rate / 100)
        
        st.metric(f"Projected Revenue Saved", f"₹{saved_amount:,.0f}", f"Assuming {save_rate}% Success")
        
        st.progress(save_rate / 100)
        
    with r2:
        st.subheader("🎯 Churn Risk Matrix")
        st.caption("X: Inactivity (Recency) | Y: Purchase Frequency. **Top Right = Lost High Value**.")
        
        fig_matrix = px.scatter(
            df, x="Recency", y="Frequency", 
            color="Lifecycle_Stage", size="LTV",
            hover_data=["Customer", "Churn_Risk_Score"],
            color_discrete_map={
                 "✨ New Customer": "#3498db", "⭐ Loyal": "#8e44ad", 
                 "🟢 Active": "#2ecc71", "⚠️ At Risk": "#f39c12", 
                 "🔴 Churned": "#e74c3c"
            }
        )
        fig_matrix.add_vline(x=90, line_dash="dash", line_color="gray", annotation_text="Churn Line")
        st.plotly_chart(fig_matrix, use_container_width=True)

    # --- 4. DETAILED REPORTING ---
    st.subheader("📋 Priority Action List")
    st.caption("Customers sorted by **Risk Score (0-100)**. Higher score = Higher urgency.")
    
    # Filter & Sort
    view_df = df[['Customer', 'Lifecycle_Stage', 'Churn_Risk_Score', 'Recency', 'LTV', 'Fav_Product']]
    view_df = view_df.sort_values('Churn_Risk_Score', ascending=False)
    
    # Formatting for display
    st.dataframe(
        view_df.style.background_gradient(subset=['Churn_Risk_Score'], cmap="Reds"),
        use_container_width=True
    )
    
    # Excel Export
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        view_df.to_excel(writer, sheet_name='Churn_Risk', index=False)
    
    st.download_button("📥 Download Priority List (Excel)", output.getvalue(), "churn_risk_analysis.xlsx")

def render_ai():
    st.title("🤖 AI Retention Strategist")
    
    if st.session_state.churn_df is None:
        st.warning("Please upload data in the Dashboard first.")
        return
        
    df = st.session_state.churn_df
    # Filter for At Risk Only
    targets = df[df['Lifecycle_Stage'] == "⚠️ At Risk"].sort_values('LTV', ascending=False)
    
    if targets.empty:
        st.success("No At-Risk customers found! Keep up the good work.")
        return

    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("Select Target")
        sel = st.selectbox("High Value At-Risk Customers", targets['Customer'])
        data = targets[targets['Customer'] == sel].iloc[0]
        
        st.markdown(f"""
        <div style='background:#fff; padding:20px; border-radius:10px; border:1px solid #ddd;'>
            <h3>{sel}</h3>
            <p><b>Risk Score:</b> {data['Churn_Risk_Score']:.1f}/100 🚨</p>
            <p><b>Absent:</b> {data['Recency']} Days</p>
            <p><b>Value:</b> ₹{data['LTV']:,.0f}</p>
            <p><b>Loves:</b> {data['Fav_Product']}</p>
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        st.subheader("AI Strategy Generator")
        action = st.radio("Strategy Type", ["💰 Discount Offer", "💬 Survey/Feedback", "🎁 VIP Status Upgrade"])
        
        if st.button("✨ Generate Plan", use_container_width=True):
            if not groq_client: st.error("Missing AI API Key"); return
            
            with st.spinner("Analyzing behavioral psychology..."):
                prompt = f"""
                Act as a Customer Success Manager.
                Customer: {sel}
                Risk Score: {data['Churn_Risk_Score']}/100 (Very High).
                Absent: {data['Recency']} days.
                Past Value: ₹{data['LTV']}.
                Favorite Product: {data['Fav_Product']}.
                
                Goal: Prevent Churn using strategy '{action}'.
                
                Output:
                1. Diagnosis: Why are they leaving? (1 sentence inference).
                2. Offer: Specific retention deal.
                3. Email: A warm, personalized subject line and body.
                """
                
                res = groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant", messages=[{"role": "user", "content": prompt}]
                )
                st.info("Strategic Recommendation:")
                st.markdown(res.choices[0].message.content)

# ==========================================
# 6. MAIN ROUTING
# ==========================================
def main():
    if not st.session_state.logged_in:
        render_login()
    else:
        with st.sidebar:
            st.title("RetainIQ")
            st.caption(f"User: {st.session_state.user}")
            st.divider()
            if st.button("📊 Dashboard"): go_to("📊 Dashboard")
            if st.button("🤖 AI Strategist"): go_to("🤖 AI Strategist")
            st.divider()
            if st.button("Logout"): logout_user()

        if st.session_state.feature == "📊 Dashboard": render_dashboard()
        else: render_ai()

if __name__ == "__main__":
    main()
