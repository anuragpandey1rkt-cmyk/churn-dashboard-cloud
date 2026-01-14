import streamlit as st
import pandas as pd
import psycopg2
import plotly.express as px
from groq import Groq
import io
import hashlib
import time
import datetime
import xlsxwriter

# ==========================================
# 1. CONFIGURATION & INIT
# ==========================================
st.set_page_config(
    page_title="RetainIQ: Churn Analytics",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional Dashboard Look
st.markdown("""
    <style>
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stMetric {
        background-color: #f9f9f9;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Load Secrets
try:
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
except:
    GROQ_API_KEY = None

# Initialize AI Client
try:
    groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
except:
    groq_client = None

# ==========================================
# 2. SESSION STATE MANAGEMENT
# ==========================================
def init_session_state():
    defaults = {
        "user": None,
        "user_name": None,
        "logged_in": False,
        "feature": "📊 Churn Dashboard",
        "churn_df": None,
        "analysis_date": None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

def go_to(page):
    st.session_state.feature = page

# ==========================================
# 3. BACKEND HELPERS (Auth & DB)
# ==========================================

def get_db_connection():
    try:
        return psycopg2.connect(
            host=st.secrets["DB_HOST"],
            database=st.secrets["DB_NAME"],
            user=st.secrets["DB_USER"],
            password=st.secrets["DB_PASS"]
        )
    except:
        return None

def init_auth_db():
    conn = get_db_connection()
    if conn:
        try:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    email VARCHAR(255) PRIMARY KEY,
                    password_hash VARCHAR(255),
                    name VARCHAR(100)
                );
            """)
            conn.commit()
            conn.close()
        except:
            pass

init_auth_db()

def hash_password(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def login_user(email, password):
    conn = get_db_connection()
    if not conn:
        st.error("Database Connection Failed (Check Secrets)")
        return

    try:
        cur = conn.cursor()
        cur.execute("SELECT password_hash, name FROM users WHERE email = %s", (email,))
        user_data = cur.fetchone()
        conn.close()

        if user_data:
            stored_hash, name = user_data
            if stored_hash == hash_password(password):
                st.session_state.user = email
                st.session_state.user_name = name
                st.session_state.logged_in = True
                st.success(f"Login Successful! Welcome {name}.")
                time.sleep(0.5)
                st.rerun()
            else:
                st.error("Invalid Password")
        else:
            st.error("User not found")
    except Exception as e:
        st.error(f"Login Error: {e}")

def signup_user(email, password, name):
    conn = get_db_connection()
    if not conn:
        st.error("Database Connection Failed")
        return
    try:
        cur = conn.cursor()
        cur.execute("INSERT INTO users (email, password_hash, name) VALUES (%s, %s, %s)",
                    (email, hash_password(password), name))
        conn.commit()
        conn.close()
        st.success("Account Created! Please Login.")
    except:
        st.error("Email already exists or DB error.")

def logout_user():
    st.session_state.user = None
    st.session_state.logged_in = False
    st.session_state.churn_df = None
    st.rerun()

# ==========================================
# 4. CHURN LOGIC ENGINE (The Core Upgrade)
# ==========================================

def process_churn_data(df):
    """
    Converts raw transactions into Behavioral Churn Metrics.
    Logic: RFM (Recency, Frequency, Monetary)
    """
    
    # 1. Clean & Map Columns
    df.columns = df.columns.str.strip()
    col_map = {
        'OrderDate': 'Date', 'CustomerName': 'Customer', 
        'SalesAmount': 'Amount', 'Product': 'Item'
    }
    # Fail-safe mapping
    for col in df.columns:
        if 'Date' in col: col_map[col] = 'Date'
        if 'Name' in col or 'Customer' in col: col_map[col] = 'Customer'
        if 'Amount' in col or 'Sales' in col: col_map[col] = 'Amount'
        if 'Product' in col or 'Item' in col: col_map[col] = 'Item'
        
    df = df.rename(columns=col_map)
    
    # Data Cleaning
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(0)
    df['Item'] = df['Item'].fillna("Unknown")

    # 2. Set Analysis Snapshot Date
    # In real-world, this is 'Today'. In datasets, it's Max Date + 1
    snapshot_date = df['Date'].max() + datetime.timedelta(days=1)

    # 3. RFM Aggregation
    # We group by Customer to derive behavior
    churn_df = df.groupby('Customer').agg({
        'Date': lambda x: (snapshot_date - x.max()).days,  # RECENCY (Critical for Churn)
        'Customer': 'count',                                # FREQUENCY (Habit)
        'Amount': 'sum',                                    # MONETARY (LTV)
        'Item': lambda x: x.mode()[0] if not x.mode().empty else "Mix" # Preference
    }).rename(columns={
        'Date': 'Days_Since_Last_Purchase',
        'Customer': 'Total_Orders',
        'Amount': 'Total_LTV',
        'Item': 'Favorite_Product'
    })

    # 4. DEFINING CHURN STATUS (Business Logic)
    # Rule Based Classification for Defensibility
    def classify_customer(row):
        recency = row['Days_Since_Last_Purchase']
        
        if recency <= 45:
            return "🟢 Active"
        elif 45 < recency <= 90:
            return "⚠️ At Risk"
        else:
            return "🔴 Churned" # Inactive > 90 Days

    churn_df['Churn_Status'] = churn_df.apply(classify_customer, axis=1)
    
    return churn_df.reset_index(), snapshot_date

# Excel Report Generator
def generate_excel_report(df):
    output = io.BytesIO()
    workbook = xlsxwriter.Workbook(output, {'in_memory': True})
    worksheet = workbook.add_worksheet()

    # Formats
    header_fmt = workbook.add_format({'bold': True, 'bg_color': '#FF4B4B', 'font_color': 'white'})
    
    # Headers
    headers = ["Customer", "Churn Status", "Days Inactive", "Total Orders", "Lifetime Value"]
    for col, h in enumerate(headers):
        worksheet.write(0, col, h, header_fmt)
    
    # Data
    for row, data in enumerate(df.itertuples(), 1):
        worksheet.write(row, 0, data.Customer)
        worksheet.write(row, 1, data.Churn_Status)
        worksheet.write(row, 2, data.Days_Since_Last_Purchase)
        worksheet.write(row, 3, data.Total_Orders)
        worksheet.write(row, 4, data.Total_LTV)
            
    workbook.close()
    return output.getvalue()

# ==========================================
# 5. UI RENDERERS
# ==========================================

def render_login_page():
    st.markdown("<h1 style='text-align: center;'>📉 RetainIQ: Churn Analytics</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Behavioral Customer Churn Prediction System</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        tab1, tab2 = st.tabs(["Login", "Sign Up"])
        with tab1:
            email = st.text_input("Email", key="l_e")
            pwd = st.text_input("Password", type="password", key="l_p")
            if st.button("Login", use_container_width=True): login_user(email, pwd)
        with tab2:
            ne = st.text_input("Email", key="s_e")
            nn = st.text_input("Name", key="s_n")
            np = st.text_input("Password", type="password", key="s_p")
            if st.button("Create Account", use_container_width=True): signup_user(ne, np, nn)

def render_dashboard():
    st.title("📊 Customer Churn Dashboard")
    st.markdown("### 📥 Input Data")
    
    uploaded_file = st.file_uploader("Upload Transaction CSV/Excel", type=['csv', 'xlsx'])
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df_raw = pd.read_csv(uploaded_file)
            else:
                df_raw = pd.read_excel(uploaded_file)
            
            # Process Data
            churn_df, analysis_date = process_churn_data(df_raw)
            st.session_state.churn_df = churn_df
            st.success(f"Churn Analysis Updated: {analysis_date.date()}")
            
        except Exception as e:
            st.error(f"Error processing file: {e}")

    # --- CHURN ANALYTICS ---
    if st.session_state.churn_df is not None:
        df = st.session_state.churn_df
        
        # 1. Churn KPIs
        st.markdown("---")
        st.subheader("🛑 High-Level Churn Metrics")
        
        total_customers = len(df)
        churned_custs = df[df['Churn_Status'] == "🔴 Churned"]
        risk_custs = df[df['Churn_Status'] == "⚠️ At Risk"]
        active_custs = df[df['Churn_Status'] == "🟢 Active"]
        
        churn_rate = (len(churned_custs) / total_customers) * 100
        rev_at_risk = risk_custs['Total_LTV'].sum() + churned_custs['Total_LTV'].sum()
        
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Churn Rate", f"{churn_rate:.1f}%", "Target: <15%", delta_color="inverse")
        kpi2.metric("Revenue at Risk", f"₹{rev_at_risk:,.0f}", "LTV of Churned/Risk Users")
        kpi3.metric("Churned Customers", len(churned_custs), "Inactive > 90 Days")
        kpi4.metric("Active Customers", len(active_custs), "Bought < 45 Days ago")
        
        st.markdown("---")

        # 2. CHURN RISK MATRIX (The Main Visual)
        c1, c2 = st.columns([2, 1])
        
        with c1:
            st.subheader("📉 Churn Risk Matrix")
            st.caption("Identify habit breakage. High Frequency + High Recency (Top Right) = **High Value Loss**.")
            
            # 
            
            fig = px.scatter(
                df, 
                x="Days_Since_Last_Purchase", 
                y="Total_Orders", # Frequency implies habit
                color="Churn_Status",
                size="Total_LTV", # Bubble size = Value
                hover_data=["Customer", "Favorite_Product"],
                color_discrete_map={
                    "🟢 Active": "green",
                    "⚠️ At Risk": "orange",
                    "🔴 Churned": "red"
                },
                labels={"Days_Since_Last_Purchase": "Days Inactive (Recency)", "Total_Orders": "Purchase Frequency"},
                title="Recency vs Frequency (Bubble Size = LTV)"
            )
            # Add Threshold Lines
            fig.add_vline(x=90, line_dash="dash", line_color="red", annotation_text="Churn Threshold")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.subheader("📊 Inactivity Histogram")
            st.caption("Distribution of days since last purchase.")
            fig_hist = px.histogram(
                df, 
                x="Days_Since_Last_Purchase",
                nbins=20,
                color="Churn_Status",
                color_discrete_map={
                    "🟢 Active": "green",
                    "⚠️ At Risk": "orange",
                    "🔴 Churned": "red"
                }
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        # 3. ACTIONABLE REPORT
        st.subheader("📋 Priority Retention List")
        st.caption("Customers marked 'At Risk' or 'Churned' sorted by Lifetime Value.")
        
        risk_filter = df[df['Churn_Status'].isin(["⚠️ At Risk", "🔴 Churned"])]
        display_cols = ['Customer', 'Churn_Status', 'Days_Since_Last_Purchase', 'Total_LTV', 'Favorite_Product']
        
        st.dataframe(
            risk_filter[display_cols].sort_values('Total_LTV', ascending=False),
            use_container_width=True,
            hide_index=True
        )
        
        # Download
        excel_data = generate_excel_report(df)
        st.download_button(
            "📥 Download Churn Analysis Report",
            data=excel_data,
            file_name="churn_analysis_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

def render_ai_consultant():
    st.title("🤖 AI Retention Consultant")
    st.markdown("Generates explainable, context-aware retention strategies based on behavioral churn data.")
    
    if st.session_state.churn_df is None:
        st.warning("Please upload data in the Dashboard first.")
        return
        
    df = st.session_state.churn_df
    # Only show risky customers
    risky_custs = df[df['Churn_Status'].isin(["⚠️ At Risk", "🔴 Churned"])]
    
    if risky_custs.empty:
        st.success("No At-Risk customers found! Great job.")
        return
        
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Select Customer")
        sel_cust = st.selectbox("Customer Name", risky_custs['Customer'].unique())
        
        cust_data = risky_custs[risky_custs['Customer'] == sel_cust].iloc[0]
        
        # Explainable Metrics
        st.error(f"Status: {cust_data['Churn_Status']}")
        st.metric("Days Inactive", f"{cust_data['Days_Since_Last_Purchase']} Days", "Churn Driver")
        st.metric("Purchase Frequency", f"{cust_data['Total_Orders']} Orders", "Habit Strength")
        st.metric("Lifetime Value", f"₹{cust_data['Total_LTV']:,.0f}")
        
    with col2:
        st.subheader("AI Diagnosis & Strategy")
        
        if st.button("✨ Generate Retention Plan", use_container_width=True):
            if not groq_client:
                st.error("AI API Key Missing")
            else:
                with st.spinner("Analyzing behavioral patterns..."):
                    try:
                        # Context-Aware Prompt
                        prompt = f"""
                        You are a Churn Analysis Expert.
                        Analyze this customer: {sel_cust}.
                        
                        Behavioral Data:
                        - Status: {cust_data['Churn_Status']} (Inactive for {cust_data['Days_Since_Last_Purchase']} days).
                        - Value: ₹{cust_data['Total_LTV']} (Lifetime Spend).
                        - Frequency: {cust_data['Total_Orders']} orders.
                        - Favorite Item: {cust_data['Favorite_Product']}.
                        
                        Task:
                        1. DIAGNOSIS: Explain WHY they are considered churned/at-risk based on the data.
                        2. STRATEGY: Suggest a specific retention offer (Discount/Bundle/Check-in).
                        3. ACTION: Draft a short, empathetic email to win them back.
                        """
                        
                        res = groq_client.chat.completions.create(
                            model="llama-3.1-8b-instant",
                            messages=[{"role": "user", "content": prompt}]
                        )
                        st.markdown(res.choices[0].message.content)
                        
                    except Exception as e:
                        st.error(f"AI Error: {e}")

# ==========================================
# 6. MAIN ROUTING
# ==========================================

def main():
    if not st.session_state.logged_in:
        render_login_page()
        return

    with st.sidebar:
        st.title("RetainIQ")
        st.caption(f"User: {st.session_state.user_name}")
        st.divider()
        if st.button("📊 Churn Dashboard", use_container_width=True): go_to("📊 Churn Dashboard")
        if st.button("🤖 AI Consultant", use_container_width=True): go_to("🤖 AI Consultant")
        st.divider()
        if st.button("Logout"): logout_user()

    if st.session_state.feature == "📊 Churn Dashboard":
        render_dashboard()
    elif st.session_state.feature == "🤖 AI Consultant":
        render_ai_consultant()

if __name__ == "__main__":
    main()
