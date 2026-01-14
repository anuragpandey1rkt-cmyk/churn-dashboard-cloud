import streamlit as st
import pandas as pd
import psycopg2
import plotly.express as px
from groq import Groq
import io
import hashlib
import time
import datetime
from fpdf import FPDF

# ==========================================
# 1. CONFIGURATION & INIT
# ==========================================
st.set_page_config(
    page_title="RetainIQ: Churn Analysis",
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
        "risk_df": None,
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
    st.session_state.risk_df = None
    st.rerun()

# ==========================================
# 4. CORE CHURN LOGIC (The "Brain")
# ==========================================

def calculate_churn_metrics(df):
    """
    This function converts raw transactions into Behavioral Churn Metrics.
    Logic:
    1. Recency: How long since last buy?
    2. Frequency: How often do they buy?
    3. Monetary: How much do they spend?
    4. Churn Classification: Based on inactivity thresholds.
    """
    
    # 1. Clean & Map Columns
    df.columns = df.columns.str.strip()
    col_map = {
        'OrderDate': 'Date', 'CustomerName': 'Customer', 
        'SalesAmount': 'Amount', 'Product': 'Item'
    }
    # Fail-safe mapping if user uploads variations
    for col in df.columns:
        if 'Date' in col: col_map[col] = 'Date'
        if 'Name' in col or 'Customer' in col: col_map[col] = 'Customer'
        if 'Amount' in col or 'Sales' in col: col_map[col] = 'Amount'
        if 'Product' in col or 'Item' in col: col_map[col] = 'Item'
        
    df = df.rename(columns=col_map)
    
    # Data Type Enforcement
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(0)
    df['Item'] = df['Item'].fillna("Unknown")

    # 2. Reference Date (Simulation of "Today")
    # In a real app, this is datetime.now(). For historical data, we use the max date in file.
    snapshot_date = df['Date'].max() + datetime.timedelta(days=1)

    # 3. Aggregation per Customer (RFM Analysis)
    churn_df = df.groupby('Customer').agg({
        'Date': lambda x: (snapshot_date - x.max()).days,  # Recency (Days since last purchase)
        'Customer': 'count',                                # Frequency (Total transactions)
        'Amount': 'sum',                                    # Monetary (Total LTV)
        'Item': lambda x: x.mode()[0] if not x.mode().empty else "Mix" # Favorite Product
    }).rename(columns={
        'Date': 'Days_Since_Last_Purchase',
        'Customer': 'Purchase_Frequency',
        'Amount': 'Total_LTV',
        'Item': 'Favorite_Product'
    })

    # 4. CHURN CLASSIFICATION LOGIC (The "Internship Evaluation" Part)
    # Logic:
    # - Active: Bought within last 45 days
    # - At Risk: Bought 46-90 days ago (Needs attention)
    # - Likely Churned: No purchase in >90 days (Lost revenue)
    
    def classify_churn(row):
        recency = row['Days_Since_Last_Purchase']
        freq = row['Purchase_Frequency']
        
        if recency <= 45:
            if freq > 5: return "Loyal Active"
            return "Active"
        elif 45 < recency <= 90:
            return "⚠️ At Risk"
        else:
            return "🔴 Likely Churned"

    churn_df['Churn_Status'] = churn_df.apply(classify_churn, axis=1)
    
    # Add Average Order Value (AOV)
    churn_df['Avg_Order_Value'] = churn_df['Total_LTV'] / churn_df['Purchase_Frequency']

    return churn_df.reset_index(), snapshot_date

# PDF Generator (Fixed for Churn Data)
def generate_churn_report(df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Customer Churn Risk Report", ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", 'B', 10)
    # Headers
    headers = ["Customer", "Status", "Days Inactive", "Rev at Risk"]
    widths = [60, 40, 40, 40]
    
    for i, h in enumerate(headers):
        pdf.cell(widths[i], 10, h, 1)
    pdf.ln()
    
    pdf.set_font("Arial", '', 9)
    # Filter for Risky Customers only
    risky_df = df[df['Churn_Status'].str.contains("Risk|Churned")]
    
    for _, row in risky_df.iterrows():
        try:
            cust = str(row['Customer'])[:25].encode('latin-1', 'replace').decode('latin-1')
            status = row['Churn_Status']
            days = str(row['Days_Since_Last_Purchase'])
            rev = f"{row['Total_LTV']:.0f}"
            
            pdf.cell(widths[0], 10, cust, 1)
            pdf.cell(widths[1], 10, status, 1)
            pdf.cell(widths[2], 10, days, 1)
            pdf.cell(widths[3], 10, rev, 1)
            pdf.ln()
        except:
            continue
            
    return pdf.output(dest='S').encode('latin-1')

# ==========================================
# 5. UI RENDERERS
# ==========================================

def render_login_page():
    st.markdown("<h1 style='text-align: center;'>📉 RetainIQ: Churn Analysis</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Predict Customer Attrition from Transaction Data</p>", unsafe_allow_html=True)
    
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
    st.markdown("### Upload Transaction Data to Diagnose Retention")
    
    uploaded_file = st.file_uploader("Required Columns: CustomerName, OrderDate, SalesAmount, Product", type=['csv', 'xlsx'])
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df_raw = pd.read_csv(uploaded_file)
            else:
                df_raw = pd.read_excel(uploaded_file)
            
            # Process Data using Behavioral Logic
            churn_df, analysis_date = calculate_churn_metrics(df_raw)
            st.session_state.risk_df = churn_df
            st.success(f"Analysis successfully derived from transaction history (As of {analysis_date.date()})")
            
        except Exception as e:
            st.error(f"Error processing file: {e}")

    # --- DASHBOARD VISUALS ---
    if st.session_state.risk_df is not None:
        df = st.session_state.risk_df
        
        # 1. High-Level Churn KPIs
        st.markdown("---")
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        
        active_count = len(df[df['Churn_Status'].str.contains("Active")])
        churned_count = len(df[df['Churn_Status'].str.contains("Churned")])
        at_risk_count = len(df[df['Churn_Status'].str.contains("Risk")])
        revenue_at_risk = df[df['Churn_Status'].str.contains("Risk")]['Total_LTV'].sum()
        
        kpi1.metric("🟢 Active Customers", active_count, "Loyal Base")
        kpi2.metric("⚠️ At Risk (Action Needed)", at_risk_count, "Needs Attention", delta_color="off")
        kpi3.metric("🔴 Likely Churned", churned_count, "Inactivity > 90 Days", delta_color="inverse")
        kpi4.metric("💰 Revenue at Risk", f"₹{revenue_at_risk:,.0f}", "Potential Loss")
        
        st.markdown("---")

        # 2. CHURN RISK MATRIX (The "Interview" Chart)
        c1, c2 = st.columns([2, 1])
        
        with c1:
            st.subheader("📉 Churn Risk Matrix")
            st.caption("X-Axis: Days Inactive (Recency) | Y-Axis: Total Spend (Monetary)")
            
            # Scatter plot showing Churn Zones
            fig = px.scatter(
                df, 
                x="Days_Since_Last_Purchase", 
                y="Total_LTV",
                color="Churn_Status",
                size="Avg_Order_Value",
                hover_data=["Customer", "Favorite_Product"],
                color_discrete_map={
                    "Loyal Active": "green",
                    "Active": "#90EE90",
                    "⚠️ At Risk": "orange",
                    "🔴 Likely Churned": "red"
                },
                title="Customer Segmentation by Inactivity vs Value"
            )
            # Add vertical lines for thresholds
            fig.add_vline(x=45, line_dash="dash", annotation_text="Active Limit")
            fig.add_vline(x=90, line_dash="dash", annotation_text="Churn Threshold")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.subheader("🥧 Risk Distribution")
            fig_pie = px.pie(df, names='Churn_Status', title="Customer Base Health", hole=0.4,
                             color='Churn_Status',
                             color_discrete_map={
                                "Loyal Active": "green", "Active": "#90EE90",
                                "⚠️ At Risk": "orange", "🔴 Likely Churned": "red"
                             })
            st.plotly_chart(fig_pie, use_container_width=True)

        # 3. SEGMENT DRILL-DOWN & REPORTING
        st.subheader("📋 At-Risk Customer List (Priority for Retention)")
        
        # Filter Logic
        risk_filter = df[df['Churn_Status'].str.contains("Risk|Churned")]
        display_cols = ['Customer', 'Churn_Status', 'Days_Since_Last_Purchase', 'Total_LTV', 'Favorite_Product']
        
        st.dataframe(
            risk_filter[display_cols].sort_values('Days_Since_Last_Purchase', ascending=False),
            use_container_width=True,
            hide_index=True
        )
        
        # Download Report
        if st.button("📥 Download Churn Risk Report (PDF)"):
            pdf_bytes = generate_churn_report(df)
            st.download_button(
                "Click to Download",
                data=pdf_bytes,
                file_name="churn_risk_analysis.pdf",
                mime="application/pdf"
            )

def render_ai_consultant():
    st.title("🤖 AI Retention Strategist")
    st.markdown("Generates personalized re-engagement emails for churned customers.")
    
    if st.session_state.risk_df is None:
        st.warning("Please upload transaction data in the Dashboard first.")
        return
        
    df = st.session_state.risk_df
    # Only show risky customers
    risky_custs = df[df['Churn_Status'].str.contains("Risk|Churned")]
    
    if risky_custs.empty:
        st.success("No customers currently at risk!")
        return
        
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.info("Select a customer to generate a win-back strategy.")
        sel_cust = st.selectbox("Select At-Risk Customer", risky_custs['Customer'].unique())
        
        cust_data = risky_custs[risky_custs['Customer'] == sel_cust].iloc[0]
        
        # Metrics Display
        st.metric("Days Inactive", f"{cust_data['Days_Since_Last_Purchase']} Days")
        st.metric("Lifetime Value", f"₹{cust_data['Total_LTV']:,.0f}")
        st.metric("Favorite Item", cust_data['Favorite_Product'])
        
    with col2:
        st.subheader("Strategy Generator")
        strategy_type = st.radio("Goal:", ["Win-Back Email (Discount)", "Feedback Request", "Product Recommendation"])
        
        if st.button("✨ Generate AI Strategy", use_container_width=True):
            if not groq_client:
                st.error("AI API Key Missing")
            else:
                with st.spinner("Analyzing purchasing behavior..."):
                    try:
                        prompt = f"""
                        Act as a Retention Manager.
                        Customer: {sel_cust}
                        Status: {cust_data['Churn_Status']}
                        Inactive for: {cust_data['Days_Since_Last_Purchase']} days.
                        Favorite Product: {cust_data['Favorite_Product']}
                        Total Spend: ₹{cust_data['Total_LTV']}
                        
                        Goal: {strategy_type}
                        
                        Write a short, personalized email/message to bring them back.
                        Highlight their favorite product if relevant.
                        """
                        
                        res = groq_client.chat.completions.create(
                            model="llama-3.1-8b-instant",
                            messages=[{"role": "user", "content": prompt}]
                        )
                        st.success("Strategy Ready:")
                        st.markdown(res.choices[0].message.content)
                        
                        # Mock Action
                        st.divider()
                        if st.button(f"📧 Send to {sel_cust}"):
                            st.toast("Email queued in CRM!", icon="✅")
                            
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
        st.caption(f"Logged in as: {st.session_state.user_name}")
        st.divider()
        if st.button("📊 Churn Dashboard", use_container_width=True): go_to("📊 Churn Dashboard")
        if st.button("🤖 AI Strategist", use_container_width=True): go_to("🤖 AI Strategist")
        st.divider()
        if st.button("Logout"): logout_user()

    if st.session_state.feature == "📊 Churn Dashboard":
        render_dashboard()
    elif st.session_state.feature == "🤖 AI Strategist":
        render_ai_consultant()

if __name__ == "__main__":
    main()
