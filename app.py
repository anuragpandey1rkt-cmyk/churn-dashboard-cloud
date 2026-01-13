import streamlit as st
import pandas as pd
import psycopg2
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from groq import Groq
import io

# ========================================================
# 1. APP CONFIGURATION & STYLING
# ========================================================
st.set_page_config(
    page_title="RetainIQ: Customer Success AI", 
    page_icon="🧠", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Real World" Look
st.markdown("""
    <style>
    .metric-card {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #fff;
        border-radius: 4px;
        color: #555;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e6f3ff;
        color: #0066cc;
    }
    </style>
""", unsafe_allow_html=True)

# ========================================================
# 2. SYSTEM FUNCTIONS
# ========================================================

# Secure Database Connection
def get_connection():
    try:
        return psycopg2.connect(
            host=st.secrets["DB_HOST"],
            database=st.secrets["DB_NAME"],
            user=st.secrets["DB_USER"],
            password=st.secrets["DB_PASS"]
        )
    except Exception as e:
        return None

# Initialize Groq AI
try:
    groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    ai_status = "✅ Active"
except:
    groq_client = None
    ai_status = "❌ Inactive (Missing Key)"

# Load Data (Cached for Speed)
@st.cache_data(ttl=600)
def load_data():
    conn = get_connection()
    if not conn:
        return None
    try:
        df = pd.read_sql("SELECT * FROM telecom_churn", conn)
        conn.close()
        return df
    except:
        return None

# Train/Retrain Model on the Fly
def train_model(df):
    le = LabelEncoder()
    # Handle variations in 'contract' column if data changes
    if 'contract' in df.columns:
        df['contract_code'] = le.fit_transform(df['contract'])
    else:
        df['contract_code'] = 0 # Fallback
        
    # Standardize Churn Column
    if df['churn'].dtype == 'object':
        df['churn_code'] = df['churn'].apply(lambda x: 1 if x == 'Yes' else 0)
    else:
        df['churn_code'] = df['churn']
    
    # Train Model (Simple Random Forest)
    X = df[['tenure', 'monthly_charges', 'contract_code']]
    y = df['churn_code']
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    return model, le

# ========================================================
# 3. SIDEBAR (SYSTEM STATUS)
# ========================================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/8649/8649607.png", width=80)
    st.title("RetainIQ 🧠")
    st.caption("Intelligent Customer Success")
    st.markdown("---")
    
    # System Health Check
    st.subheader("System Status")
    
    # DB Check
    conn_check = get_connection()
    if conn_check:
        st.success(f"Database: Connected")
        conn_check.close()
    else:
        st.error("Database: Disconnected")
        
    # AI Check
    if groq_client:
        st.success(f"AI Engine: {ai_status}")
    else:
        st.warning(f"AI Engine: {ai_status}")
        
    st.markdown("---")
    st.info("💡 **Pro Tip:** Upload your weekly sales Excel file in the 'Batch Predict' tab to forecast risks.")

# ========================================================
# 4. MAIN APPLICATION
# ========================================================

# Load Data
df_main = load_data()

if df_main is None:
    st.error("🚨 Critical Error: Could not connect to Supabase Database. Please check your Secrets.")
    st.stop()

# Train Model
model, le = train_model(df_main)

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Executive Dashboard", "📂 Batch Risk Analysis", "🤖 AI Consultant"])

# --------------------------------------------------------
# TAB 1: EXECUTIVE DASHBOARD
# --------------------------------------------------------
with tab1:
    st.header("Real-Time Retention Overview")
    
    # KPIs
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Total Active Customers", f"{len(df_main):,}")
    with kpi2:
        churn_rate = df_main['churn_code'].mean() * 100
        st.metric("Current Churn Rate", f"{churn_rate:.1f}%", delta="-0.5%" if churn_rate < 30 else "+1.2%", delta_color="inverse")
    with kpi3:
        avg_rev = df_main['monthly_charges'].mean()
        st.metric("Avg. Revenue Per User", f"${avg_rev:.2f}")
    with kpi4:
        risk_count = len(df_main[df_main['churn_code']==1])
        st.metric("High Risk Customers", f"{risk_count}", "Action Needed", delta_color="inverse")

    st.markdown("---")

    # Interactive Charts
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📉 Churn Distribution by Contract")
        # Enhancing the visual
        fig_contract = px.bar(df_main.groupby(['contract', 'churn']).size().reset_index(name='count'), 
                              x="contract", y="count", color="churn", 
                              color_discrete_map={'Yes':'#FF4B4B', 'No':'#2E8B57'},
                              barmode='group')
        st.plotly_chart(fig_contract, use_container_width=True)
    
    with c2:
        st.subheader("💰 Revenue Risk Analysis")
        fig_scatter = px.scatter(df_main, x="tenure", y="monthly_charges", color="churn",
                               size="monthly_charges", 
                               color_discrete_map={'Yes':'#FF4B4B', 'No':'#2E8B57'},
                               hover_data=['customer_id'])
        st.plotly_chart(fig_scatter, use_container_width=True)

# --------------------------------------------------------
# TAB 2: BATCH PREDICTION (EXCEL + CSV)
# --------------------------------------------------------
with tab2:
    st.header("📂 Bulk Risk Analysis")
    st.write("Upload your weekly customer data (Excel or CSV) to identify at-risk accounts instantly.")
    
    # 1. Template Download
    col_dl, col_up = st.columns([1, 2])
    with col_dl:
        st.info("📋 **Required Format:**\nColumns: `tenure`, `monthly_charges`, `contract`")
        # Create a mock excel file in memory for download
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            pd.DataFrame({
                'tenure': [12, 5, 72],
                'monthly_charges': [70.5, 95.0, 110.0],
                'contract': ['Month-to-month', 'Month-to-month', 'Two year']
            }).to_excel(writer, index=False)
        
        st.download_button(
            label="⬇️ Download Excel Template",
            data=buffer.getvalue(),
            file_name="customer_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    # 2. File Uploader
    uploaded_file = st.file_uploader("Upload File", type=['csv', 'xlsx'])
    
    if uploaded_file is not None:
        try:
            # Detect File Type and Load
            if uploaded_file.name.endswith('.csv'):
                df_upload = pd.read_csv(uploaded_file)
            else:
                df_upload = pd.read_excel(uploaded_file)
            
            st.write(f"✅ Loaded **{len(df_upload)} rows** from `{uploaded_file.name}`")
            
            # Validation: Check if columns exist
            required_cols = ['tenure', 'monthly_charges', 'contract']
            missing_cols = [col for col in required_cols if col not in df_upload.columns]
            
            if missing_cols:
                st.error(f"⚠️ Missing Columns: {', '.join(missing_cols)}")
                st.write("Please rename your columns to match the template.")
            else:
                # 3. RUN PREDICTION
                if st.button("🚀 Analyze Risk Factors"):
                    with st.spinner("AI Model is processing..."):
                        # Preprocess
                        df_upload['contract_code'] = le.transform(df_upload['contract'].astype(str))
                        X_new = df_upload[['tenure', 'monthly_charges', 'contract_code']]
                        
                        # Predict
                        probs = model.predict_proba(X_new)[:, 1]
                        df_upload['Churn Probability'] = probs
                        df_upload['Risk Status'] = ["🔴 High Risk" if p > 0.5 else "🟢 Safe" for p in probs]
                        
                        # Show Results with Color
                        st.dataframe(
                            df_upload.style.applymap(
                                lambda v: 'color: red; font-weight: bold;' if 'High Risk' in str(v) else 'color: green;', 
                                subset=['Risk Status']
                            )
                        )
                        
                        # Download Results
                        res_buffer = io.BytesIO()
                        with pd.ExcelWriter(res_buffer, engine='openpyxl') as writer:
                            df_upload.to_excel(writer, index=False)
                        
                        st.download_button(
                            "⬇️ Download Analysis Report (.xlsx)",
                            data=res_buffer.getvalue(),
                            file_name="RetainIQ_Analysis_Report.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                        
        except Exception as e:
            st.error(f"Error processing file: {e}")

# --------------------------------------------------------
# TAB 3: AI CONSULTANT
# --------------------------------------------------------
with tab3:
    st.header("🤖 Intelligent Retention Consultant")
    
    c_left, c_right = st.columns([1, 2])
    
    with c_left:
        st.subheader("Select Customer")
        # Filter high risk
        risk_df = df_main[df_main['churn_code'] == 1]
        if risk_df.empty:
            st.success("🎉 No high-risk customers found!")
        else:
            selected_cust_str = st.selectbox(
                "Choose an At-Risk Profile:",
                risk_df['customer_id'].astype(str) + " (" + risk_df['contract'] + ")"
            )
            cust_id = selected_cust_str.split(" (")[0]
            cust_data = df_main[df_main['customer_id'].astype(str) == cust_id].iloc[0]
            
            st.info(f"""
            **Profile:**
            - 🆔 ID: {cust_id}
            - ⏱️ Tenure: {cust_data['tenure']} months
            - 💵 Bill: ${cust_data['monthly_charges']}
            - 📜 Contract: {cust_data['contract']}
            """)

    with c_right:
        st.subheader("AI Strategy Generator")
        if risk_df.empty:
            st.write("System is all clear.")
        else:
            action = st.radio("What do you want to do?", ["Analyze Why", "Draft Retention Email", "Offer Calculation"])
            
            if st.button("✨ Generate AI Response"):
                if not groq_client:
                    st.error("Please configure GROQ_API_KEY in Secrets.")
                else:
                    with st.spinner("Consulting Llama-3 AI..."):
                        
                        prompt_context = f"""
                        Customer Profile:
                        - Tenure: {cust_data['tenure']} months
                        - Monthly Bill: ${cust_data['monthly_charges']}
                        - Contract: {cust_data['contract']}
                        - Risk Level: High
                        """
                        
                        if action == "Analyze Why":
                            prompt = f"{prompt_context}\nExplain 3 psychological reasons why this specific customer might be churning based on their contract and bill."
                        elif action == "Draft Retention Email":
                            prompt = f"{prompt_context}\nWrite a short, warm, professional email offering a 15% discount if they renew for 1 year."
                        else:
                            prompt = f"{prompt_context}\nCalculate the Lifetime Value (LTV) if they stay for 12 more months vs if they leave now."

                        completion = groq_client.chat.completions.create(
                            model="llama3-8b-8192",
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.7
                        )
                        
                        st.success("Strategy Ready:")
                        st.markdown(completion.choices[0].message.content)
