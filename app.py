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
# 1. CONFIGURATION & PROFESSIONAL UI
# ==========================================
st.set_page_config(
    page_title="RetainIQ: Customer Churn Analytics",
    page_icon="📉",
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
    h1, h2, h3 { color: #0f172a; font-family: 'Helvetica Neue', sans-serif; }
    </style>
""", unsafe_allow_html=True)

# Load Secrets safely
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", None)
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# ==========================================
# 2. SESSION STATE & AUTH HELPERS
# ==========================================
if 'user' not in st.session_state:
    st.session_state.update({
        "user": None, "logged_in": False, "feature": "📊 Dashboard", "churn_df": None, "raw_df": None
    })

def go_to(page): st.session_state.feature = page

def get_db_connection():
    try:
        return psycopg2.connect(
            host=st.secrets["DB_HOST"], database=st.secrets["DB_NAME"],
            user=st.secrets["DB_USER"], password=st.secrets["DB_PASS"]
        )
    except: return None

def hash_password(password): return hashlib.sha256(str.encode(password)).hexdigest()

def login_user(email, password):
    # 1. Admin/Demo Fallback (Prioritize this to skip DB logic)
    if email == "admin" and password == "admin":
        st.session_state.user = "admin"
        st.session_state.user_name = "Evaluator"
        st.session_state.logged_in = True
        st.success("Login Successful! Loading Dashboard...")
        time.sleep(0.5)
        st.rerun()
        return  # CRITICAL: Stop execution here to prevent falling into DB errors

    # 2. Database Connection Check
    conn = get_db_connection()
    if not conn:
        st.error("Database unavailable. Use demo credentials (admin/admin).")
        return

    # 3. Standard Authentication
    try:
        cur = conn.cursor()
        cur.execute("SELECT password_hash, name FROM users WHERE email = %s", (email,))
        data = cur.fetchone()
        conn.close() # Close connection immediately after fetching

        # Verify Password
        if data and data[0] == hash_password(password):
            st.session_state.user = email
            st.session_state.user_name = data[1]
            st.session_state.logged_in = True
            st.success(f"Welcome back, {data[1]}!")
            time.sleep(0.5)
            st.rerun()
        else:
            st.error("Invalid email or password.")

    except Exception as e:
        st.error(f"Login Error: {e}")
        if conn: conn.close()
            
def signup_user(email, password, name):
    conn = get_db_connection()
    if not conn:
        st.error("Database unavailable. Use demo credentials.")
        return
    try:
        cur = conn.cursor()
        # Create table if it doesn't exist (Safety check)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                email VARCHAR(255) PRIMARY KEY, 
                password_hash VARCHAR(255), 
                name VARCHAR(100)
            );
        """)
        # Insert new user
        cur.execute(
            "INSERT INTO users (email, password_hash, name) VALUES (%s, %s, %s)",
            (email, hash_password(password), name)
        )
        conn.commit()
        conn.close()
        st.success("✅ Account Created! Please go to the Login tab.")
    except Exception as e:
        st.error(f"Signup Error: {e} (Email might already exist)")
        
def logout_user():
    st.session_state.clear()
    st.rerun()

# ==========================================
# 3. ADVANCED CHURN ENGINE (Logic Core)
# ==========================================

def process_advanced_analytics(df):
    """
    CONVERTS TRANSACTIONS TO BEHAVIORAL CHURN METRICS.
    
    Logic Definition:
    1. Snapshot Date: The day after the dataset's last transaction.
    2. Recency: Days since last purchase.
    3. Churn Threshold: >90 Days (Retail Standard).
    4. Risk Threshold: 45-90 Days.
    5. Churn Risk Score (0-100): Composite of Recency (70%) and Frequency (30%).
    """
    
    # --- 1. Robust Column Mapping ---
    df.columns = df.columns.str.strip()
    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if 'date' in cl: col_map[c] = 'Date'
        elif 'name' in cl or 'customer' in cl: col_map[c] = 'Customer'
        elif 'amount' in cl or 'sales' in cl: col_map[c] = 'Amount'
        elif 'product' in cl or 'item' in cl: col_map[c] = 'Item'
    
    df = df.rename(columns=col_map)
    
    # --- 2. Defensive Data Cleaning ---
    # Coerce errors to NaT/NaN and drop them to prevent crashes
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(0)
    df.dropna(subset=['Date', 'Customer'], inplace=True)
    
    if df.empty:
        return None, None, None

    # Snapshot Date (Simulation of "Today")
    snapshot_date = df['Date'].max() + datetime.timedelta(days=1)

    # --- 3. RFM Aggregation ---
    metrics = df.groupby('Customer').agg({
        'Date': lambda x: (snapshot_date - x.max()).days, # Recency
        'Customer': 'count',                              # Frequency
        'Amount': 'sum',                                  # Monetary (LTV)
        'Item': lambda x: x.mode()[0] if not x.mode().empty else "Mix"
    }).rename(columns={
        'Date': 'Recency', 
        'Customer': 'Frequency', 
        'Amount': 'LTV',
        'Item': 'Fav_Product'
    })
    
    # --- 4. Churn Risk Scoring (0-100) ---
    # Normalize Recency (Cap at 120 days): Older = Higher Score (Bad)
    metrics['Recency_Score'] = metrics['Recency'].apply(lambda x: min(x, 120) / 120 * 100)
    
    # Normalize Frequency: Lower Frequency = Higher Score (Bad)
    # 1 order = 100 risk, 10+ orders = 0 risk component
    metrics['Freq_Score'] = metrics['Frequency'].apply(lambda x: max(0, 100 - (x * 10)))
    
    # Composite Score: 70% Recency weight (primary churn driver), 30% Frequency
    metrics['Churn_Risk_Score'] = (0.7 * metrics['Recency_Score']) + (0.3 * metrics['Freq_Score'])
    
    # --- 5. Lifecycle Segmentation (Hierarchical Logic) ---
    def define_lifecycle(row):
        recency = row['Recency']
        freq = row['Frequency']
        
        # Priority 1: Churn Status (The most critical health metric)
        if recency > 90: return "🔴 Churned"
        if recency > 45: return "⚠️ At Risk"
        
        # Priority 2: Loyalty (For active users)
        if freq == 1: return "✨ New Customer"
        if freq >= 5: return "⭐ Loyal"
        
        return "🟢 Active"

    metrics['Lifecycle_Stage'] = metrics.apply(define_lifecycle, axis=1)
    
    return metrics.reset_index(), df, snapshot_date

# ==========================================
# 4. DASHBOARD UI
# ==========================================

def render_login():
    st.markdown("<div style='text-align: center; padding: 40px;'><h1>📉 RetainIQ</h1><p>Customer Churn Analysis Suite</p></div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        # Create Tabs to switch between Login and Sign Up
        tab1, tab2 = st.tabs(["Login", "Create Account"])
        
        # --- TAB 1: LOGIN ---
        with tab1:
            with st.form("login_form"):
                e = st.text_input("Email", key="login_email")
                p = st.text_input("Password", type="password", key="login_pass")
                if st.form_submit_button("Access Dashboard", use_container_width=True):
                    login_user(e, p)
            st.caption("Evaluators: Use `admin` / `admin` if DB is offline.")

        # --- TAB 2: SIGN UP ---
        with tab2:
            st.markdown("### New User?")
            with st.form("signup_form"):
                new_email = st.text_input("Email", key="signup_email")
                new_name = st.text_input("Full Name", key="signup_name")
                new_pass = st.text_input("Password", type="password", key="signup_pass")
                if st.form_submit_button("Register Now", use_container_width=True):
                    signup_user(new_email, new_pass, new_name)
                    
def render_dashboard():
    st.title("📊 Customer Churn Analysis")
    st.markdown("Diagnose retention health using behavioral transaction data.")
    
    # --- FILE UPLOAD ---
    uploaded_file = st.file_uploader("Upload Data (Cols: CustomerName, OrderDate, SalesAmount, Product)", type=['csv', 'xlsx'])
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'): raw = pd.read_csv(uploaded_file)
            else: raw = pd.read_excel(uploaded_file)
            
            churn_df, raw_clean, snap_date = process_advanced_analytics(raw)
            
            if churn_df is None:
                st.error("Data Validation Failed: Ensure columns exist and contain valid data.")
            else:
                st.session_state.churn_df = churn_df
                st.session_state.raw_df = raw_clean
                
        except Exception as e: st.error(f"File Error: {e}")

    if st.session_state.churn_df is None:
        st.info("👋 Upload transaction data to generate the churn report.")
        return

    df = st.session_state.churn_df
    raw_df = st.session_state.raw_df
    
    # --- 1. EXECUTIVE KPIS ---
    st.markdown("### 🛑 Retention Health")
    k1, k2, k3, k4 = st.columns(4)
    
    # Calculate Metrics
    total_cust = len(df)
    churn_cnt = len(df[df['Lifecycle_Stage'] == "🔴 Churned"])
    risk_cnt = len(df[df['Lifecycle_Stage'] == "⚠️ At Risk"])
    rev_at_risk = df[df['Lifecycle_Stage'].isin(["🔴 Churned", "⚠️ At Risk"])]['LTV'].sum()
    churn_rate = (churn_cnt / total_cust) * 100 if total_cust > 0 else 0
    
    k1.metric("Total Customers", total_cust)
    k2.metric("Churn Rate", f"{churn_rate:.1f}%", "-Target: <15%", delta_color="inverse")
    k3.metric("At Risk Customers", risk_cnt, "Immediate Action Required", delta_color="off")
    k4.metric("Revenue at Risk", f"₹{rev_at_risk:,.0f}", "LTV of Risk/Churned")
    
    st.markdown("---")

    # --- 2. CHURN DIAGNOSTICS (Charts) ---
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.subheader("📉 Churn Risk Matrix")
        st.caption("Visualizing the relationship between Inactivity (Recency) and Value (Frequency).")
        
        # Explicit Churn Matrix
        fig_matrix = px.scatter(
            df, x="Recency", y="Frequency", 
            color="Lifecycle_Stage", size="LTV",
            hover_data=["Customer", "Churn_Risk_Score"],
            color_discrete_map={
                 "✨ New Customer": "#3498db", "⭐ Loyal": "#8e44ad", 
                 "🟢 Active": "#2ecc71", "⚠️ At Risk": "#f39c12", 
                 "🔴 Churned": "#e74c3c"
            },
            labels={"Recency": "Days Since Last Purchase (Risk)", "Frequency": "Purchase Count (Habit)"},
            title="Customer Distribution"
        )
        # Add Churn Threshold Line
        fig_matrix.add_vline(x=90, line_dash="dash", line_color="red", annotation_text="Churn > 90 Days")
        st.plotly_chart(fig_matrix, use_container_width=True)
        
    with c2:
        st.subheader("👥 Lifecycle Split")
        st.caption("Current status of customer base.")
        fig_pie = px.pie(df, names='Lifecycle_Stage', hole=0.5, 
                         color='Lifecycle_Stage',
                         color_discrete_map={
                             "✨ New Customer": "#3498db", "⭐ Loyal": "#8e44ad", 
                             "🟢 Active": "#2ecc71", "⚠️ At Risk": "#f39c12", 
                             "🔴 Churned": "#e74c3c"
                         })
        fig_pie.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig_pie, use_container_width=True)

    # --- 3. TREND ANALYSIS ---
    st.subheader("🗓️ Monthly Churn Trend")
    st.caption("Are we losing or gaining active customers over time?")
    
    # Calculate MAU (Monthly Active Users)
    raw_df['Month'] = raw_df['Date'].dt.to_period('M').astype(str)
    mau = raw_df.groupby('Month')['Customer'].nunique().reset_index()
    fig_trend = px.line(mau, x='Month', y='Customer', markers=True, 
                        title="Monthly Active Customers (MAU)")
    fig_trend.update_traces(line_color='#1E3A8A', line_width=3)
    st.plotly_chart(fig_trend, use_container_width=True)

    # --- 4. PRIORITY ACTION LIST (THE FIX) ---
    st.subheader("📋 Priority Action List")
    st.caption("Customers with highest Churn Risk Score (0-100). Focus retention efforts here.")
    
    view_df = df[['Customer', 'Lifecycle_Stage', 'Churn_Risk_Score', 'Recency', 'LTV', 'Fav_Product']]
    view_df = view_df.sort_values('Churn_Risk_Score', ascending=False).head(50) # Top 50 risky
    
    # FIXED: Using st.column_config instead of pandas styler to prevent Streamlit Cloud crashes
    st.dataframe(
        view_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Churn_Risk_Score": st.column_config.ProgressColumn(
                "Risk Score",
                help="0=Safe, 100=Critical Risk",
                format="%d",
                min_value=0,
                max_value=100,
            ),
            "LTV": st.column_config.NumberColumn(
                "Lifetime Value",
                format="₹%d"
            ),
            "Recency": st.column_config.NumberColumn(
                "Days Inactive",
                format="%d days"
            )
        }
    )
    
    # Excel Export
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        view_df.to_excel(writer, sheet_name='Churn_Risk', index=False)
    st.download_button("📥 Download Full Report (Excel)", output.getvalue(), "churn_risk_analysis.xlsx")

def render_ai():
    st.title("🤖 AI Retention Consultant")
    st.markdown("Generate personalized win-back strategies for high-risk customers.")
    
    # 1. Check for Data
    if st.session_state.churn_df is None:
        st.warning("Upload data in the Dashboard first.")
        return
        
    df = st.session_state.churn_df
    # Filter: High Value AND At Risk/Churned
    targets = df[df['Lifecycle_Stage'].isin(["⚠️ At Risk", "🔴 Churned"])].sort_values('LTV', ascending=False)
    
    if targets.empty:
        st.success("No High-Risk customers found! Retention is healthy.")
        return

    # 2. Top Section: Customer Selection & Strategy Generator
    c1, c2 = st.columns([1, 2])
    
    with c1:
        st.subheader("Select Customer")
        sel = st.selectbox("High-Value At Risk", targets['Customer'])
        data = targets[targets['Customer'] == sel].iloc[0]
        
        st.info(f"""
        **Profile:** {sel}
        - **Stage:** {data['Lifecycle_Stage']}
        - **Risk Score:** {data['Churn_Risk_Score']:.0f}/100
        - **Value:** ₹{data['LTV']:,.0f}
        - **Absent:** {data['Recency']} Days
        """)
        
    with c2:
        st.subheader("Strategy Generator")
        action = st.radio("Intervention Type", ["🎁 Win-Back Offer", "💬 Feedback Request", "📦 Product Bundle"])
        
        if st.button("✨ Generate Strategy"):
            if not groq_client: 
                st.error("AI API Key Missing in Secrets")
            else:
                with st.spinner("Analyzing churn vectors..."):
                    prompt_strategy = f"""
                    You are a Retention Expert.
                    Customer: {sel}
                    Stage: {data['Lifecycle_Stage']} (Inactive {data['Recency']} days).
                    Value: ₹{data['LTV']}.
                    Favorite Product: {data['Fav_Product']}.
                    Risk Score: {data['Churn_Risk_Score']}/100.
                    
                    Goal: Prevent churn using '{action}'.
                    
                    Output:
                    1. **Diagnosis**: Why are they at risk? (Based on inactivity/habit).
                    2. **Offer**: A specific financial or value-based incentive.
                    3. **Communication**: A short, empathetic email draft.
                    """
                    try:
                        res = groq_client.chat.completions.create(
                            model="llama-3.1-8b-instant", messages=[{"role": "user", "content": prompt_strategy}]
                        )
                        st.success("Strategic Recommendation:")
                        st.markdown(res.choices[0].message.content)
                    except Exception as e:
                        st.error(f"AI Error: {e}")

    # ==========================================
    # 3. CHAT INTERFACE (OUTSIDE COLUMNS & BUTTONS)
    # ==========================================
    # This must be dedented to the main function level
    
    st.divider()
    st.subheader(f"💬 Chat about {sel}")
    st.caption("Ask follow-up questions to refine your retention strategy.")

    # Initialize Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Reset Chat if Customer Changes
    if "last_selected_cust" not in st.session_state or st.session_state.last_selected_cust != sel:
        st.session_state.messages = []
        st.session_state.last_selected_cust = sel

    # Display History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle Input
    # Note: Using a different variable name 'chat_input' to avoid conflict with strategy 'prompt'
    # 4. Handle New User Input
    if chat_input := st.chat_input(f"Ask about {sel}..."):
        # Display User Message
        st.session_state.messages.append({"role": "user", "content": chat_input})
        with st.chat_message("user"):
            st.markdown(chat_input)

        # Generate AI Response
        with st.chat_message("assistant"):
            if not groq_client:
                st.error("AI Key Missing")
            else:
                try:
                    # Context for the AI
                    system_context = f"""
                    You are a Retention Expert.
                    Focus Customer: {sel}
                    - Status: {data['Lifecycle_Stage']}
                    - Days Inactive: {data['Recency']}
                    - Total Value: ₹{data['LTV']}
                    - Favorite Product: {data['Fav_Product']}
                    - Risk Score: {data['Churn_Risk_Score']}/100
                    
                    Answer the user's question specifically about this customer.
                    Keep answers concise and text-only (no JSON).
                    """
                    
                    # Create the stream
                    stream = groq_client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=[
                            {"role": "system", "content": system_context},
                            # Filter out any bad history to prevent crashes
                            *[m for m in st.session_state.messages if isinstance(m.get('content'), str)]
                        ],
                        stream=True
                    )
                    
                    # --- THE FIX: Generator to extract ONLY text ---
                    def clean_stream():
                        for chunk in stream:
                            # Extract text content safely
                            if chunk.choices[0].delta.content:
                                yield chunk.choices[0].delta.content

                    # Write the clean text to screen
                    response_text = st.write_stream(clean_stream())
                    
                    # Save ONLY the text to history
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                    
                except Exception as e:
                    st.error(f"Chat Error: {e}")
                    
                    
# ==========================================
# 5. MAIN ROUTING
# ==========================================
def main():
    if not st.session_state.logged_in:
        render_login()
    else:
        with st.sidebar:
            st.title("RetainIQ")
            st.caption(f"Logged in: {st.session_state.user}")
            st.divider()
            if st.button("📊 Churn Dashboard"): go_to("📊 Dashboard")
            if st.button("🤖 AI Consultant"): go_to("🤖 AI Consultant")
            st.divider()
            if st.button("Logout"): logout_user()

        if st.session_state.feature == "📊 Dashboard": render_dashboard()
        else: render_ai()

if __name__ == "__main__":
    main()
