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
    page_title="RetainIQ Pro",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional Look
st.markdown("""
    <style>
    .metric-card {background-color: #f0f2f6; border-radius: 10px; padding: 15px; border-left: 5px solid #4B4BFF;}
    .risk-high {color: #FF4B4B; font-weight: bold;}
    .risk-safe {color: #2E8B57; font-weight: bold;}
    .block-container {padding-top: 2rem; padding-bottom: 5rem;}
    </style>
""", unsafe_allow_html=True)

# Load Secrets safely
try:
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", None)
except:
    GROQ_API_KEY = None

# Initialize AI Client
try:
    if GROQ_API_KEY:
        groq_client = Groq(api_key=GROQ_API_KEY)
    else:
        groq_client = None
except:
    groq_client = None

# ==========================================
# 2. SESSION STATE MANAGEMENT
# ==========================================
def init_session_state():
    defaults = {
        "user": None,          # Stores user email/id
        "user_name": None,     # Stores user display name
        "logged_in": False,    # Auth status
        "feature": "📊 Dashboard", # Current active page
        "risk_df": None,       # Stores analyzed data
        "analysis_date": None  # Stores date of analysis
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# Navigation Helper
def go_to(page):
    st.session_state.feature = page

# ==========================================
# 3. BACKEND HELPERS (Auth, DB, Logic)
# ==========================================

# --- DATABASE CONNECTION ---
def get_db_connection():
    try:
        return psycopg2.connect(
            host=st.secrets["DB_HOST"],
            database=st.secrets["DB_NAME"],
            user=st.secrets["DB_USER"],
            password=st.secrets["DB_PASS"]
        )
    except Exception as e:
        return None

# --- AUTHENTICATION (SQL Based) ---
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

init_auth_db() # Run once on startup

def hash_password(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def login_user(email, password):
    conn = get_db_connection()
    if not conn:
        st.error("❌ Database Connection Failed. Check Secrets.")
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
                st.success(f"Welcome back, {name}!")
                time.sleep(1)
                st.rerun()
            else:
                st.error("Incorrect Password")
        else:
            st.error("User not found")
    except Exception as e:
        st.error(f"Login Error: {e}")

def signup_user(email, password, name):
    conn = get_db_connection()
    if not conn:
        st.error("❌ Database Connection Failed. Check Secrets.")
        return

    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO users (email, password_hash, name) VALUES (%s, %s, %s)",
            (email, hash_password(password), name)
        )
        conn.commit()
        conn.close()
        st.success("✅ Account Created! Please Login.")
    except psycopg2.errors.UniqueViolation:
        st.error("Email already exists.")
    except Exception as e:
        st.error(f"Signup Error: {e}")

def logout_user():
    st.session_state.user = None
    st.session_state.logged_in = False
    st.session_state.risk_df = None
    st.rerun()

# --- CORE BUSINESS LOGIC (Smart RFM Analysis) ---
def process_data(df):
    # 1. Clean Headers (Remove spaces)
    df.columns = df.columns.str.strip()
    
    # 2. Smart Column Search
    # Matches: OrderDate/Date, CustomerName/Customer, SalesAmount/Amount
    col_map = {}
    
    # Helper to find column containing a keyword
    def find_col(keywords):
        for col in df.columns:
            for kw in keywords:
                if kw.lower() in col.lower():
                    return col
        return None

    # Find Columns
    col_map[find_col(['OrderDate', 'Date', 'Time'])] = 'Date'
    col_map[find_col(['Customer', 'Name', 'Client'])] = 'Customer'
    col_map[find_col(['Sales', 'Amount', 'Price', 'Value'])] = 'Amount'
    col_map[find_col(['Product', 'Item', 'SKU'])] = 'Item'

    # Remove None values from map
    col_map = {k: v for k, v in col_map.items() if k is not None}

    # Rename
    df = df.rename(columns=col_map)
    
    # 3. Validation
    required = ['Date', 'Customer', 'Amount']
    missing = [col for col in required if col not in df.columns]
    
    if missing:
        return None, f"⚠️ Column Error! We found: {list(df.columns)}. We need: {required}"
        
    # 4. Processing (Fail-Safe)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(0)
    
    if 'Item' not in df.columns:
        df['Item'] = "General Item"
    else:
        df['Item'] = df['Item'].fillna("Unknown")

    # 5. RFM Analysis
    if df['Date'].isna().all():
         return None, "⚠️ Error: Date column parsing failed. Check date format."

    snapshot_date = df['Date'].max()
    
    # Safe Mode Aggregation
    rfm = df.groupby('Customer').agg({
        'Date': lambda x: (snapshot_date - x.max()).days,
        'Customer': 'count',
        'Amount': 'sum',
        'Item': lambda x: x.mode()[0] if not x.mode().empty else "Mix"
    }).rename(columns={'Date': 'Days_Silent', 'Customer': 'Orders', 'Amount': 'Total_Spent', 'Item': 'Top_Item'})
    
    # 6. Risk Scoring
    def get_risk(row):
        if row['Days_Silent'] > 90: return 'High Risk'
        elif row['Days_Silent'] > 45: return 'Medium Risk'
        else: return 'Safe'
    
    rfm['Status'] = rfm.apply(get_risk, axis=1)
    return rfm.reset_index(), None

def generate_excel(df_risk):
    # Crash-Proof Excel Generator
    output = io.BytesIO()
    workbook = xlsxwriter.Workbook(output, {'in_memory': True})
    worksheet = workbook.add_worksheet()

    # Formats
    header_fmt = workbook.add_format({'bold': True, 'bg_color': '#4B4BFF', 'font_color': 'white'})
    money_fmt = workbook.add_format({'num_format': '[$₹-4009]#,##0'}) # Native Excel Rupee Support
    
    # Write Headers
    headers = ['Customer', 'Days Silent', 'Total Spent', 'Top Item', 'Status']
    for col, h in enumerate(headers):
        worksheet.write(0, col, h, header_fmt)
        
    # Write Data
    for row, data in enumerate(df_risk.itertuples(), 1):
        if data.Status != 'Safe':
            worksheet.write(row, 0, data.Customer)
            worksheet.write(row, 1, data.Days_Silent)
            worksheet.write(row, 2, data.Total_Spent, money_fmt)
            worksheet.write(row, 3, data.Top_Item)
            worksheet.write(row, 4, data.Status)
            
    workbook.close()
    return output.getvalue()
    
# ==========================================
# 4. FEATURE RENDERERS
# ==========================================

def render_login_page():
    st.markdown("<h1 style='text-align: center;'>🧠 RetainIQ Pro</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Intelligent Customer Retention System</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        tab1, tab2 = st.tabs(["Login", "Create Account"])
        
        with tab1:
            email = st.text_input("Email", key="l_email")
            password = st.text_input("Password", type="password", key="l_pass")
            if st.button("Login", use_container_width=True):
                if email and password:
                    login_user(email, password)
                else:
                    st.warning("Please fill all fields")
        
        with tab2:
            new_email = st.text_input("Email", key="s_email")
            new_name = st.text_input("Full Name", key="s_name")
            new_pass = st.text_input("Password", type="password", key="s_pass")
            if st.button("Sign Up", use_container_width=True):
                if new_email and new_name and new_pass:
                    signup_user(new_email, new_pass, new_name)
                else:
                    st.warning("Please fill all fields")

def render_dashboard():
    st.header("📊 Transaction Analyzer")
    st.info("Upload CSV/Excel with columns: `CustomerName`, `OrderDate`, `SalesAmount`, `Product`")
    
    uploaded_file = st.file_uploader("Upload Sales Data", type=['csv', 'xlsx'])
    
    # Process Upload
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df_raw = pd.read_csv(uploaded_file)
            else:
                df_raw = pd.read_excel(uploaded_file)
            
            risk_df, error = process_data(df_raw)
            
            if error:
                st.error(error)
            else:
                # Store in Session State
                st.session_state.risk_df = risk_df
                st.session_state.analysis_date = datetime.date.today()
                st.success("✅ Analysis Complete!")
        except Exception as e:
            st.error(f"File Error: {e}")

    # Display Results (if data exists)
    if st.session_state.risk_df is not None:
        risk_df = st.session_state.risk_df
        
        # Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Customers", len(risk_df))
        high_risk = len(risk_df[risk_df['Status']=='High Risk'])
        m2.metric("🚨 High Risk", high_risk)
        total_rev = risk_df['Total_Spent'].sum()
        m3.metric("💰 Total Revenue", f"₹{total_rev:,.0f}")
        avg_val = risk_df['Total_Spent'].mean()
        m4.metric("Avg Spend", f"₹{avg_val:,.0f}")
        
        st.divider()
        
        # Charts & Data
        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader("📉 Risk Matrix")
            fig = px.scatter(risk_df, x="Days_Silent", y="Total_Spent",
                                color="Status", size="Total_Spent",
                                hover_data=['Customer', 'Top_Item'],
                                color_discrete_map={'High Risk':'red', 'Medium Risk':'orange', 'Safe':'green'},
                                labels={'Total_Spent': 'Total Spend (₹)', 'Days_Silent': 'Days Since Last Order'})
            st.plotly_chart(fig, use_container_width=True)
            
        with c2:
            st.subheader("📋 Risk Report")
            # Display Table
            display_df = risk_df[risk_df['Status'].str.contains('Risk')][['Customer', 'Top_Item', 'Total_Spent', 'Status']].sort_values('Total_Spent', ascending=False)
            
            # Add Rupee Symbol for Display Only
            display_df['Total_Spent'] = display_df['Total_Spent'].apply(lambda x: f"₹{x:,.0f}")
            
            st.dataframe(display_df, hide_index=True)
            
            # Generate Excel
            excel_data = generate_excel(risk_df)
                
            st.download_button(
                label="⬇️ Download Risk Report (Excel)",
                data=excel_data,
                file_name="Risk_Report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            ) 
            
def render_ai_consultant():
    st.header("🤖 AI Retention Specialist")
    
    if st.session_state.risk_df is None:
        st.warning("⚠️ Please go to the Dashboard and upload data first.")
        return

    risk_df = st.session_state.risk_df
    # Filter for risks
    risky_custs = risk_df[risk_df['Status'].str.contains('Risk')]
    
    if risky_custs.empty:
        st.success("✅ No high-risk customers found!")
        return

    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Select Customer")
        sel_cust = st.selectbox("Choose Customer", risky_custs['Customer'].unique())
        
        # Get Data Safely
        cust_data = risky_custs[risky_custs['Customer'] == sel_cust].iloc[0]
        
        # Safe variables for Display & AI
        c_days = cust_data['Days_Silent']
        c_val = f"{cust_data['Total_Spent']:,.0f}"
        c_item = str(cust_data['Top_Item']) # Force string to prevent crashes
        
        st.markdown(f"""
        <div style='background-color:#f0f2f6; padding:15px; border-radius:10px;'>
            <h4>{sel_cust}</h4>
            <p><b>Status:</b> {cust_data['Status']}</p>
            <p><b>Silent:</b> {c_days} days</p>
            <p><b>Value:</b> ₹{c_val}</p>
            <p><b>Loves:</b> {c_item}</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.subheader("Generate Strategy")
        action = st.radio("Choose Action", ["Analyze Churn Reason", "Draft Email Offer", "Calculate LTV"])
        
        if st.button("✨ Ask AI", use_container_width=True):
            if not groq_client:
                st.error("❌ AI Key Missing in Secrets. Please add GROQ_API_KEY.")
            else:
                with st.spinner("Consulting AI..."):
                    try:
                        # Safe Prompt Construction
                        prompt = f"""
                        Role: Senior Retention Manager.
                        Task: Save Customer '{sel_cust}'.
                        
                        Customer Data:
                        - Absent: {c_days} days.
                        - Value: {c_val} INR.
                        - Favorite: {c_item}.
                        
                        Action Required: {action}
                        
                        Output:
                        1. Diagnosis (1 sentence).
                        2. Win-back Offer (using ₹ symbol).
                        3. Email Draft (Professional & Warm).
                        """
                        
                        res = groq_client.chat.completions.create(
                            model="llama-3.1-8b-instant",
                            messages=[{"role": "user", "content": prompt}]
                        )
                        st.success("Strategy Generated:")
                        st.markdown(res.choices[0].message.content)
                    except Exception as e:
                        st.error(f"AI Error: {e}")

# ==========================================
# 5. MAIN APP LOGIC
# ==========================================

def main():
    # 1. CHECK AUTH STATUS
    if not st.session_state.logged_in:
        render_login_page()
        return

    # 2. SIDEBAR NAVIGATION (Only if logged in)
    with st.sidebar:
        st.title("RetainIQ")
        st.write(f"👤 {st.session_state.user_name}")
        
        # Navigation Buttons
        if st.button("📊 Dashboard", use_container_width=True): go_to("📊 Dashboard")
        if st.button("🤖 AI Consultant", use_container_width=True): go_to("🤖 AI Consultant")
        
        st.divider()
        
        # System Status
        conn = get_db_connection()
        if conn:
            st.caption("🟢 Database: Connected")
            conn.close()
        else:
            st.caption("🔴 Database: Disconnected")
            
        if groq_client:
            st.caption("🟢 AI Engine: Active")
        else:
            st.caption("🔴 AI Engine: Inactive")
            
        st.divider()
        if st.button("🚪 Logout", use_container_width=True): logout_user()

    # 3. ROUTING
    page = st.session_state.feature
    
    if page == "📊 Dashboard":
        render_dashboard()
    elif page == "🤖 AI Consultant":
        render_ai_consultant()

if __name__ == "__main__":
    main()
