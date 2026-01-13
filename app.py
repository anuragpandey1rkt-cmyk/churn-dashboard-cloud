import streamlit as st
import pandas as pd
import psycopg2
import bcrypt
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import plotly.express as px

# ==========================================
# 1. APP CONFIGURATION & STYLING
# ==========================================
st.set_page_config(page_title="EcoWise AI | Enterprise Churn Analytics", layout="wide", page_icon="⚡")

# Custom CSS for Professional Look
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    .stButton>button {width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white;}
    .stMetric {background-color: white; padding: 15px; border-radius: 10px; box-shadow: 2px 2px 10px rgba(0,0,0,0.1);}
    </style>
""", unsafe_allow_html=True)

# Initialize Session State for Login
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'username' not in st.session_state:
    st.session_state['username'] = ""

# ==========================================
# 2. DATABASE FUNCTIONS
# ==========================================
def get_connection():
    return psycopg2.connect(
        host=st.secrets["DB_HOST"],
        database=st.secrets["DB_NAME"],
        user=st.secrets["DB_USER"],
        password=st.secrets["DB_PASS"]
    )

def run_query(query, params=(), fetch=False):
    conn = get_connection()
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
        st.error(f"Database Error: {e}")
        return False
    finally:
        cur.close()
        conn.close()

# ==========================================
# 3. AUTHENTICATION SYSTEM
# ==========================================
def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def check_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def login_page():
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image("https://cdn-icons-png.flaticon.com/512/2991/2991148.png", width=100)
        st.title("🔐 Enterprise Login")
        st.markdown("Access the **EcoWise AI** secure prediction platform.")
        
        tab1, tab2 = st.tabs(["Login", "Sign Up"])
        
        with tab1:
            login_user = st.text_input("Username", key="l_user")
            login_pass = st.text_input("Password", type="password", key="l_pass")
            if st.button("Log In"):
                user_data = run_query("SELECT password_hash FROM app_users WHERE username = %s", (login_user,), fetch=True)
                if not user_data.empty:
                    stored_hash = user_data.iloc[0]['password_hash']
                    if check_password(login_pass, stored_hash):
                        st.session_state['logged_in'] = True
                        st.session_state['username'] = login_user
                        st.success("Login Successful!")
                        st.rerun()
                    else:
                        st.error("Incorrect Password")
                else:
                    st.error("User not found.")

        with tab2:
            new_user = st.text_input("Create Username", key="s_user")
            new_pass = st.text_input("Create Password", type="password", key="s_pass")
            if st.button("Create Account"):
                if new_user and new_pass:
                    hashed = hash_password(new_pass)
                    success = run_query("INSERT INTO app_users (username, password_hash) VALUES (%s, %s)", (new_user, hashed))
                    if success:
                        st.success("Account created! Please log in.")
                    else:
                        st.error("Username already exists.")

# ==========================================
# 4. DASHBOARD & AI ENGINE
# ==========================================
@st.cache_data
def train_model():
    df = run_query("SELECT * FROM telecom_churn", fetch=True)
    if df.empty: return None, None, None
    
    le = LabelEncoder()
    df['contract_code'] = le.fit_transform(df['contract'])
    df['churn_code'] = df['churn'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    X = df[['tenure', 'monthly_charges', 'contract_code']]
    y = df['churn_code']
    
    model = RandomForestClassifier()
    model.fit(X, y)
    return model, le, df

def dashboard():
    st.sidebar.title(f"👤 Welcome, {st.session_state['username']}")
    menu = st.sidebar.radio("Navigation", ["📊 Dashboard Overview", "🤖 AI Prediction", "📥 Add Customer Data", "📁 Batch Analysis"])
    
    if st.sidebar.button("Logout"):
        st.session_state['logged_in'] = False
        st.rerun()

    model, le, df = train_model()
    
    if menu == "📊 Dashboard Overview":
        st.title("📊 Executive Insights")
        
        # KPI ROW
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Customers", len(df))
        churn_rate = df['churn_code'].mean() * 100
        c2.metric("Churn Rate", f"{churn_rate:.1f}%", delta="-2%" if churn_rate < 30 else "+5%", delta_color="inverse")
        avg_rev = df['monthly_charges'].mean()
        c3.metric("Avg Revenue", f"${avg_rev:.2f}")
        at_risk = len(df[df['churn'] == 'Yes'])
        c4.metric("Customers at Risk", at_risk, "Needs Attention")

        # CHARTS ROW
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Revenue Distribution")
            fig = px.histogram(df, x="monthly_charges", nbins=20, title="Monthly Charges Spread", color_discrete_sequence=['#636EFA'])
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.subheader("Churn by Contract")
            fig2 = px.pie(df, names='contract', values='churn_code', title="Risk by Contract Type", hole=0.4)
            st.plotly_chart(fig2, use_container_width=True)

    elif menu == "🤖 AI Prediction":
        st.title("🤖 Real-Time Risk Analysis")
        st.markdown("Use this tool to predict if a specific customer is about to leave.")
        
        col1, col2, col3 = st.columns(3)
        p_tenure = col1.slider("Tenure (Months)", 1, 72, 12)
        p_charges = col2.number_input("Monthly Charges ($)", 10, 200, 70)
        p_contract = col3.selectbox("Contract Type", le.classes_)
        
        if st.button("Analyze Customer Risk"):
            contract_val = le.transform([p_contract])[0]
            prob = model.predict_proba([[p_tenure, p_charges, contract_val]])[0][1]
            
            st.markdown("### 🔍 Prediction Result")
            if prob > 0.6:
                st.error(f"🚨 **HIGH RISK** ({prob*100:.1f}% Probability of Churn)")
                st.info("💡 Recommendation: Offer a 15% discount on 1-year contract renewal.")
            elif prob > 0.3:
                st.warning(f"⚠️ **MODERATE RISK** ({prob*100:.1f}% Probability)")
            else:
                st.success(f"✅ **SAFE** ({prob*100:.1f}% Probability)")

    elif menu == "📥 Add Customer Data":
        st.title("📥 Enter New Data")
        with st.form("entry_form"):
            c1, c2 = st.columns(2)
            tenure = c1.number_input("Tenure (Months)", 1, 72)
            monthly = c2.number_input("Monthly Charges", 10.0, 200.0)
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            churn_status = st.selectbox("Current Status", ["No", "Yes"])
            
            if st.form_submit_button("Save to Database"):
                query = "INSERT INTO telecom_churn (tenure, monthly_charges, contract, churn) VALUES (%s, %s, %s, %s)"
                success = run_query(query, (tenure, monthly, contract, churn_status))
                if success:
                    st.success("Data Saved Successfully!")
                    time.sleep(1)
                    st.rerun()

    elif menu == "📁 Batch Analysis":
        st.title("📁 Bulk Prediction")
        st.markdown("Upload a CSV file to predict churn for hundreds of customers at once.")
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file:
            batch_df = pd.read_csv(uploaded_file)
            st.write("Preview:", batch_df.head())
            if st.button("Run Batch Prediction"):
                # Preprocessing for batch
                batch_df['contract_code'] = le.transform(batch_df['contract'])
                X_batch = batch_df[['tenure', 'monthly_charges', 'contract_code']]
                batch_df['Churn_Probability'] = model.predict_proba(X_batch)[:, 1]
                
                st.dataframe(batch_df.style.highlight_max(axis=0, color='lightred'))
                st.download_button("Download Predictions", batch_df.to_csv(), "predictions.csv")

# ==========================================
# 5. MAIN APP FLOW
# ==========================================
if st.session_state['logged_in']:
    dashboard()
else:
    login_page()
