import streamlit as st
import pandas as pd
import psycopg2
import plotly.express as px
from groq import Groq
import io
from fpdf import FPDF
import hashlib

# ========================================================
# 1. APP CONFIGURATION
# ========================================================
st.set_page_config(page_title="RetainIQ Pro", page_icon="🔒", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .metric-card {background-color: #f0f2f6; border-radius: 10px; padding: 15px; border-left: 5px solid #4B4BFF;}
    .risk-label {font-weight: bold; padding: 2px 8px; border-radius: 4px;}
    .stButton>button {width: 100%; border-radius: 5px;}
    </style>
""", unsafe_allow_html=True)

# ========================================================
# 2. FAIL-SAFE CONNECTIONS
# ========================================================

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

try:
    groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])
except:
    groq_client = None

# ========================================================
# 3. AUTHENTICATION SYSTEM (LOGIN/SIGNUP)
# ========================================================

def init_auth_db():
    """Creates the users table if it doesn't exist"""
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
            cur.close()
            conn.close()
        except Exception as e:
            st.error(f"Auth DB Init Error: {e}")

def hash_password(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def login_user(email, password):
    conn = get_db_connection()
    if not conn:
        return False, "Database Offline"
    
    cur = conn.cursor()
    cur.execute("SELECT password_hash, name FROM users WHERE email = %s", (email,))
    user = cur.fetchone()
    conn.close()
    
    if user:
        stored_hash, name = user
        if stored_hash == hash_password(password):
            return True, name
    return False, "Invalid Email or Password"

def signup_user(email, password, name):
    conn = get_db_connection()
    if not conn:
        return False, "Database Offline"
    
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO users (email, password_hash, name) VALUES (%s, %s, %s)",
            (email, hash_password(password), name)
        )
        conn.commit()
        conn.close()
        return True, "Account Created! Please Login."
    except psycopg2.errors.UniqueViolation:
        return False, "Email already exists."
    except Exception as e:
        return False, str(e)

# Initialize Auth Table
init_auth_db()

# ========================================================
# 4. CORE LOGIC (Sales Analysis)
# ========================================================

def process_data(df):
    df.columns = df.columns.str.strip()
    
    col_map = {
        'OrderDate': 'Date', 'date': 'Date',
        'CustomerName': 'Customer', 'Customer': 'Customer', 
        'SalesAmount': 'Amount', 'Sales': 'Amount', 'Amount': 'Amount',
        'Product': 'Item', 'ProductName': 'Item', 'Item': 'Item'
    }
    df = df.rename(columns=col_map)
    
    required = ['Date', 'Customer', 'Amount']
    if not all(col in df.columns for col in required):
        return None, f"Missing columns. Found: {list(df.columns)}"
        
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(0)
    if 'Item' not in df.columns:
        df['Item'] = "General Item"

    snapshot_date = df['Date'].max()
    
    rfm = df.groupby('Customer').agg({
        'Date': lambda x: (snapshot_date - x.max()).days,
        'Customer': 'count',
        'Amount': 'sum',
        'Item': lambda x: x.mode()[0] if not x.mode().empty else "Unknown"
    }).rename(columns={'Date': 'Days_Silent', 'Customer': 'Orders', 'Amount': 'Total_Spent', 'Item': 'Top_Item'})
    
    def get_risk(row):
        if row['Days_Silent'] > 90: return 'High Risk'
        elif row['Days_Silent'] > 45: return 'Medium Risk'
        else: return 'Safe'
        
    rfm['Status'] = rfm.apply(get_risk, axis=1)
    return rfm.reset_index(), None

def generate_pdf(df_risk):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "RetainIQ Risk Report", ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(55, 10, "Customer", 1)
    pdf.cell(25, 10, "Days Silent", 1)
    pdf.cell(40, 10, "Total (Rs.)", 1)
    pdf.cell(40, 10, "Top Item", 1)
    pdf.cell(30, 10, "Status", 1)
    pdf.ln()
    
    pdf.set_font("Arial", '', 9)
    for _, row in df_risk.iterrows():
        if row['Status'] != "Safe":
            pdf.cell(55, 10, str(row['Customer'])[:25], 1)
            pdf.cell(25, 10, str(row['Days_Silent']), 1)
            pdf.cell(40, 10, f"{row['Total_Spent']:.0f}", 1)
            pdf.cell(40, 10, str(row['Top_Item'])[:20], 1)
            pdf.cell(30, 10, row['Status'], 1)
            pdf.ln()
    return pdf.output(dest='S').encode('latin-1', 'ignore')

# ========================================================
# 5. UI CONTROLLER (LOGIN VS DASHBOARD)
# ========================================================

# Check Session State
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'username' not in st.session_state:
    st.session_state['username'] = ""

# --- LOGIN SCREEN ---
if not st.session_state['logged_in']:
    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        st.title("🔒 RetainIQ Login")
        auth_mode = st.radio("Select Mode", ["Login", "Sign Up"], horizontal=True)
        
        email = st.text_input("Email Address")
        password = st.text_input("Password", type="password")
        
        if auth_mode == "Sign Up":
            name = st.text_input("Full Name")
            if st.button("Create Account"):
                if email and password and name:
                    success, msg = signup_user(email, password, name)
                    if success:
                        st.success(msg)
                    else:
                        st.error(msg)
                else:
                    st.warning("Please fill all fields.")
                    
        else: # Login Mode
            if st.button("Access Dashboard"):
                if email and password:
                    success, msg = login_user(email, password)
                    if success:
                        st.session_state['logged_in'] = True
                        st.session_state['username'] = msg
                        st.rerun()
                    else:
                        st.error(msg)
                else:
                    st.warning("Enter Email and Password")

# --- MAIN DASHBOARD (Only visible after login) ---
else:
    # Sidebar
    with st.sidebar:
        st.title(f"Welcome, {st.session_state['username']}")
        if st.button("🚪 Logout"):
            st.session_state['logged_in'] = False
            st.session_state['username'] = ""
            st.rerun()
        st.divider()
        st.header("Status Panel")
        if get_db_connection():
            st.success("Database: Online")
        else:
            st.warning("Database: Offline")
        if groq_client:
            st.success("AI Engine: Online")
        else:
            st.error("AI Engine: Offline")

    st.title("🧠 RetainIQ: Customer Analysis")
    
    tab1, tab2 = st.tabs(["📂 Upload Transaction Data", "🤖 AI Consultant"])

    # TAB 1: UPLOAD
    with tab1:
        st.info("Upload Excel/CSV with: `CustomerName`, `OrderDate`, `SalesAmount`, `Product`")
        uploaded_file = st.file_uploader("Drop your Sales File Here", type=['csv', 'xlsx'])
        
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df_raw = pd.read_csv(uploaded_file)
                else:
                    df_raw = pd.read_excel(uploaded_file)
                
                st.success(f"Loaded {len(df_raw)} transactions.")
                risk_df, error = process_data(df_raw)
                
                if error:
                    st.error(error)
                else:
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Total Customers", len(risk_df))
                    high_risk = len(risk_df[risk_df['Status']=='High Risk'])
                    m2.metric("High Risk", high_risk)
                    total_rev = risk_df['Total_Spent'].sum()
                    m3.metric("Total Revenue", f"₹{total_rev:,.0f}")
                    avg_val = risk_df['Total_Spent'].mean()
                    m4.metric("Avg Spend", f"₹{avg_val:,.0f}")
                    
                    st.divider()
                    
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
                        display_df = risk_df[risk_df['Status'].str.contains('Risk')][['Customer', 'Top_Item', 'Total_Spent', 'Status']].sort_values('Total_Spent', ascending=False)
                        display_df['Total_Spent'] = display_df['Total_Spent'].apply(lambda x: f"₹{x:,.0f}")
                        st.dataframe(display_df, hide_index=True)
                        
                        pdf_data = generate_pdf(risk_df)
                        st.download_button("⬇️ Download PDF Report", data=pdf_data, file_name="Risk_Report.pdf", mime="application/pdf")
                        
                    st.session_state['risk_df'] = risk_df
                    
            except Exception as e:
                st.error(f"File Error: {e}")

    # TAB 2: AI
    with tab2:
        if 'risk_df' not in st.session_state:
            st.warning("Please upload a file in Tab 1 first.")
        else:
            risk_df = st.session_state['risk_df']
            risky_custs = risk_df[risk_df['Status'].str.contains('Risk')]
            
            if risky_custs.empty:
                st.success("No high risk customers found.")
            else:
                sel_cust = st.selectbox("Select Customer to Recover:", risky_custs['Customer'].unique())
                cust_data = risky_custs[risky_custs['Customer'] == sel_cust].iloc[0]
                
                st.info(f"""
                **Profile:** {sel_cust}
                - 🛑 Status: {cust_data['Status']}
                - 🗓️ Last Seen: {cust_data['Days_Silent']} days ago
                - 💰 Total Value: ₹{cust_data['Total_Spent']:,.0f}
                - 🛍️ Loves: {cust_data['Top_Item']}
                """)
                
                if st.button("Generate Recovery Plan"):
                    if not groq_client:
                        st.error("AI Key missing in Secrets.")
                    else:
                        with st.spinner("Drafting strategy..."):
                            prompt = f"""
                            You are a Customer Success Manager.
                            Customer: {sel_cust}
                            Last bought: {cust_data['Days_Silent']} days ago.
                            Total Spend: ₹{cust_data['Total_Spent']}.
                            Favorite Item: {cust_data['Top_Item']}.
                            
                            1. Why might they have churned? (1 sentence)
                            2. Create a "Come Back" offer using Rupees (₹).
                            3. Draft a 3-sentence email to them.
                            """
                            res = groq_client.chat.completions.create(
                                model="llama3-8b-8192",
                                messages=[{"role": "user", "content": prompt}]
                            )
                            st.markdown(res.choices[0].message.content)
