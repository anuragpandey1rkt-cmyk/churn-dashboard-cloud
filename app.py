import streamlit as st
import pandas as pd
import psycopg2
import time
from supabase import create_client, Client
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import plotly.express as px

# ==========================================
# 1. APP CONFIGURATION
# ==========================================
st.set_page_config(page_title="EcoWise AI | Secure Login", layout="wide", page_icon="🛡️")

# Custom CSS
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    .stButton>button {width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white;}
    .stMetric {background-color: white; padding: 15px; border-radius: 10px; box-shadow: 2px 2px 10px rgba(0,0,0,0.1);}
    </style>
""", unsafe_allow_html=True)

# Initialize Session State
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'user_email' not in st.session_state:
    st.session_state['user_email'] = ""

# ==========================================
# 2. CONNECTIONS (DB + AUTH)
# ==========================================
# Connection A: For Data (Charts/SQL)
def get_db_connection():
    return psycopg2.connect(
        host=st.secrets["DB_HOST"],
        database=st.secrets["DB_NAME"],
        user=st.secrets["DB_USER"],
        password=st.secrets["DB_PASS"]
    )

# Connection B: For Auth (Login/Signup)
@st.cache_resource
def init_supabase():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

supabase = init_supabase()

# ==========================================
# 3. AUTHENTICATION (SUPABASE)
# ==========================================
def login_page():
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image("https://cdn-icons-png.flaticon.com/512/2991/2991148.png", width=100)
        st.title("🛡️ Secure Access")
        st.markdown("Login via **Supabase Auth** (Email Verification Required).")
        
        tab1, tab2 = st.tabs(["Login", "Sign Up"])
        
        # --- LOGIN TAB ---
        with tab1:
            email = st.text_input("Email", key="l_email")
            password = st.text_input("Password", type="password", key="l_pass")
            
            if st.button("Log In"):
                try:
                    # Attempt to sign in with Supabase
                    response = supabase.auth.sign_in_with_password({"email": email, "password": password})
                    st.session_state['logged_in'] = True
                    st.session_state['user_email'] = response.user.email
                    st.success("✅ Login Successful!")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    # Supabase returns specific errors (e.g. "Email not confirmed")
                    st.error(f"❌ Login Failed: {e}")
                    if "Email not confirmed" in str(e):
                        st.warning("⚠️ You must click the verification link sent to your email before logging in.")

        # --- SIGN UP TAB ---
        with tab2:
            st.info("ℹ️ We will send a verification link to your inbox.")
            new_email = st.text_input("Enter Email", key="s_email")
            new_pass = st.text_input("Create Password", type="password", key="s_pass")
            
            if st.button("Create Account"):
                try:
                    # Create user in Supabase
                    response = supabase.auth.sign_up({"email": new_email, "password": new_pass})
                    st.success(f"🎉 Account created for {new_email}!")
                    st.info("📧 **IMPORTANT:** Check your inbox now and click the verification link. You cannot login until you verify.")
                except Exception as e:
                    st.error(f"Error: {e}")

# ==========================================
# 4. DASHBOARD LOGIC (SAME AS BEFORE)
# ==========================================
def run_query(query, params=(), fetch=False):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(query, params)
        if fetch:
            result = cur.fetchall()
            columns = [desc[0] for desc in cur.description]
            df = pd.DataFrame(result, columns=columns)
            return df
        conn.commit()
        return True
    except Exception as e:
        return None
    finally:
        cur.close()
        conn.close()

@st.cache_data
def train_model():
    df = run_query("SELECT * FROM telecom_churn", fetch=True)
    if df is None or df.empty: return None, None, None
    
    le = LabelEncoder()
    df['contract_code'] = le.fit_transform(df['contract'])
    df['churn_code'] = df['churn'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    X = df[['tenure', 'monthly_charges', 'contract_code']]
    y = df['churn_code']
    
    model = RandomForestClassifier()
    model.fit(X, y)
    return model, le, df

def dashboard():
    st.sidebar.success(f"Logged in as: {st.session_state['user_email']}")
    
    if st.sidebar.button("Logout"):
        supabase.auth.sign_out()
        st.session_state['logged_in'] = False
        st.rerun()

    # --- MAIN CONTENT ---
    menu = st.sidebar.radio("Navigation", ["Overview", "AI Prediction", "Data Entry"])
    model, le, df = train_model()

    if menu == "Overview":
        st.title("📊 Live Data Overview")
        c1, c2, c3 = st.columns(3)
        c1.metric("Customers", len(df))
        c2.metric("Churn Rate", f"{(df['churn_code'].mean()*100):.1f}%")
        c3.metric("Avg Bill", f"${df['monthly_charges'].mean():.2f}")
        
        fig = px.histogram(df, x="contract", color="churn", barmode="group", title="Churn by Contract")
        st.plotly_chart(fig, use_container_width=True)

    elif menu == "AI Prediction":
        st.title("🤖 Customer Risk AI")
        c1, c2 = st.columns(2)
        tenure = c1.slider("Tenure", 1, 72, 12)
        charges = c2.number_input("Charges", 20, 150, 70)
        contract = st.selectbox("Contract", le.classes_)
        
        if st.button("Predict"):
            c_val = le.transform([contract])[0]
            prob = model.predict_proba([[tenure, charges, c_val]])[0][1]
            if prob > 0.5:
                st.error(f"High Risk: {prob:.2%}")
            else:
                st.success(f"Safe: {prob:.2%}")

    elif menu == "Data Entry":
        st.title("📥 Add New Customer")
        with st.form("add"):
            t = st.number_input("Tenure", 1, 72)
            m = st.number_input("Monthly Bill", 10, 200)
            c = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            status = st.selectbox("Churned?", ["No", "Yes"])
            if st.form_submit_button("Save"):
                run_query("INSERT INTO telecom_churn (tenure, monthly_charges, contract, churn) VALUES (%s, %s, %s, %s)", (t, m, c, status))
                st.success("Saved!")
                time.sleep(1)
                st.rerun()

# ==========================================
# 5. MAIN FLOW
# ==========================================
if st.session_state['logged_in']:
    dashboard()
else:
    login_page()
