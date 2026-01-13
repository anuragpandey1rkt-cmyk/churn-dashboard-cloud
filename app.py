import streamlit as st
import pandas as pd
import psycopg2
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from groq import Groq
import io

# 1. APP CONFIGURATION
st.set_page_config(page_title="EcoWise AI: Retention Pro", page_icon="🚀", layout="wide")

# 2. SAFER DATABASE CONNECTION
def get_connection():
    try:
        return psycopg2.connect(
            host=st.secrets["DB_HOST"],
            database=st.secrets["DB_NAME"],
            user=st.secrets["DB_USER"],
            password=st.secrets["DB_PASS"]
        )
    except Exception as e:
        st.error(f"🚨 DB Error: {e}")
        st.stop()

# 3. AI CLIENT SETUP (GROQ)
try:
    groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])
except:
    st.warning("⚠️ Groq API Key missing. AI Consultant features will be disabled.")
    groq_client = None

# 4. LOAD & CACHE DATA
@st.cache_data(ttl=600)
def load_data():
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM telecom_churn", conn)
    conn.close()
    return df

# 5. TRAIN MODEL (Re-trains instantly on load)
def train_model(df):
    le = LabelEncoder()
    df['contract_code'] = le.fit_transform(df['contract'])
    # Handle 'Yes'/'No' or 1/0 for churn
    if df['churn'].dtype == 'object':
        df['churn_code'] = df['churn'].apply(lambda x: 1 if x == 'Yes' else 0)
    else:
        df['churn_code'] = df['churn']
    
    X = df[['tenure', 'monthly_charges', 'contract_code']]
    y = df['churn_code']
    model = RandomForestClassifier()
    model.fit(X, y)
    return model, le

# Load Data
df_main = load_data()
model, le = train_model(df_main)

# ========================================================
# 🎨 UI LAYOUT: TABS FOR PROFESSIONAL FEEL
# ========================================================
st.title("🚀 EcoWise AI: Customer Retention Suite")
tab1, tab2, tab3 = st.tabs(["📊 Live Dashboard", "📂 Batch Predict (Upload)", "🤖 AI Consultant"])

# --------------------------------------------------------
# TAB 1: LIVE DASHBOARD (Real-Time Insights)
# --------------------------------------------------------
with tab1:
    st.subheader("📡 Real-Time Database Overview")
    
    # KPI Row
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Customers", len(df_main))
    churn_rate = df_main['churn_code'].mean() * 100
    kpi2.metric("Churn Rate", f"{churn_rate:.1f}%", delta_color="inverse")
    kpi3.metric("Avg Monthly Revenue", f"${df_main['monthly_charges'].mean():.2f}")
    kpi4.metric("At-Risk Customers", len(df_main[df_main['churn_code']==1]))

    # Charts Row
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Churn by Contract Type")
        fig_contract = px.histogram(df_main, x="contract", color="churn", barmode="group", 
                                  color_discrete_map={'Yes':'red', 'No':'green', 1:'red', 0:'green'})
        st.plotly_chart(fig_contract, use_container_width=True)
    
    with c2:
        st.caption("Revenue vs Tenure Risk Map")
        fig_scatter = px.scatter(df_main, x="tenure", y="monthly_charges", color="churn", 
                               color_discrete_map={'Yes':'red', 'No':'green', 1:'red', 0:'green'},
                               hover_data=['customer_id'])
        st.plotly_chart(fig_scatter, use_container_width=True)

# --------------------------------------------------------
# TAB 2: BATCH PREDICTION (Upload CSV)
# --------------------------------------------------------
with tab2:
    st.subheader("📂 Upload Customer List")
    st.write("Upload a CSV file with columns: `tenure`, `monthly_charges`, `contract`")
    
    # Template Download
    csv_template = "tenure,monthly_charges,contract\n12,70.5,Month-to-month\n72,110.0,Two year"
    st.download_button("⬇️ Download CSV Template", csv_template, "template.csv", "text/csv")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            st.success("File Uploaded Successfully!")
            
            # Predict
            if st.button("🚀 Run Prediction on All Rows"):
                # Preprocess
                df_upload['contract_code'] = le.transform(df_upload['contract'])
                X_new = df_upload[['tenure', 'monthly_charges', 'contract_code']]
                
                # Get Probabilities
                probs = model.predict_proba(X_new)[:, 1]
                df_upload['Churn Probability'] = probs
                df_upload['Prediction'] = ["High Risk" if p > 0.5 else "Safe" for p in probs]
                
                st.dataframe(df_upload.style.applymap(lambda v: 'color: red;' if v == 'High Risk' else 'color: green;', subset=['Prediction']))
                
                # Download Result
                csv_result = df_upload.to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Download Predictions", csv_result, "churn_predictions.csv", "text/csv")
                
        except Exception as e:
            st.error(f"Error parsing file: {e}")

# --------------------------------------------------------
# TAB 3: AI CONSULTANT (Groq Integration)
# --------------------------------------------------------
with tab3:
    st.subheader("🤖 AI Retention Consultant")
    st.write("Select a high-risk customer to generate a personalized retention strategy.")
    
    # Filter only High Risk customers
    risk_df = df_main[df_main['churn_code'] == 1].head(10)
    
    if risk_df.empty:
        st.info("No high-risk customers found in the database!")
    else:
        selected_cust = st.selectbox("Select At-Risk Customer", risk_df['customer_id'].astype(str) + " - " + risk_df['contract'])
        
        # Get Customer Details
        cust_id = selected_cust.split(" - ")[0]
        cust_data = df_main[df_main['customer_id'].astype(str) == cust_id].iloc[0]
        
        # Display Stats
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Tenure", f"{cust_data['tenure']} Months")
        col_b.metric("Monthly Bill", f"${cust_data['monthly_charges']}")
        col_c.metric("Contract", cust_data['contract'])
        
        if st.button("✨ Generate AI Retention Plan"):
            if not groq_client:
                st.error("Please add GROQ_API_KEY to secrets to use this feature.")
            else:
                with st.spinner("AI is analyzing customer profile..."):
                    # Prompt Engineering
                    prompt = f"""
                    You are a Retention Expert at a Telecom company.
                    A customer is at high risk of churning (leaving).
                    
                    Profile:
                    - Tenure: {cust_data['tenure']} months
                    - Monthly Bill: ${cust_data['monthly_charges']}
                    - Contract: {cust_data['contract']}
                    
                    Task:
                    1. Explain WHY they are likely to leave (briefly).
                    2. Suggest a specific offer to keep them.
                    3. Write a short, empathetic email to this customer offering the deal.
                    """
                    
                    completion = groq_client.chat.completions.create(
                        model="llama3-8b-8192",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.7,
                        max_tokens=400
                    )
                    
                    st.success("Strategy Generated!")
                    st.markdown(completion.choices[0].message.content)
