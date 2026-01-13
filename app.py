import streamlit as st
import pandas as pd
import psycopg2
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from groq import Groq
import io
from datetime import datetime
from fpdf import FPDF

# ========================================================
# 1. APP CONFIGURATION
# ========================================================
st.set_page_config(
    page_title="RetainIQ: Customer Intelligence", 
    page_icon="🧠", 
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .metric-card {background-color: #f0f2f6; border-radius: 10px; padding: 15px;}
    .risk-high {color: #FF4B4B; font-weight: bold;}
    .risk-low {color: #2E8B57; font-weight: bold;}
    </style>
""", unsafe_allow_html=True)

# ========================================================
# 2. SYSTEM FUNCTIONS
# ========================================================

def get_connection():
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
# 3. ADVANCED LOGIC: TRANSACTION TO CHURN (RFM)
# ========================================================
def process_sales_data(df):
    """
    Turns raw sales rows (Date, Customer, Amount) into Churn Risk Profile.
    """
    # 1. Standardize Columns
    # Map your specific file columns to standard names
    col_map = {
        'OrderDate': 'date', 'Date': 'date',
        'CustomerName': 'customer', 'Customer': 'customer',
        'SalesAmount': 'amount', 'Amount': 'amount', 'Price': 'amount'
    }
    df = df.rename(columns=col_map)
    
    # 2. Convert Date
    df['date'] = pd.to_datetime(df['date'])
    
    # 3. Reference Date (Today or max date in file)
    snapshot_date = df['date'].max()
    
    # 4. RFM Calculation (The Magic)
    rfm = df.groupby('customer').agg({
        'date': lambda x: (snapshot_date - x.max()).days, # Recency (Days since last buy)
        'customer': 'count', # Frequency (Count of orders)
        'amount': 'sum' # Monetary (Total spend)
    }).rename(columns={'date': 'Recency', 'customer': 'Frequency', 'amount': 'Monetary'})
    
    # 5. Risk Rule Engine (Customize this logic!)
    # Rule: If haven't bought in 90 days = High Risk
    # Rule: If bought recently but spent little = Medium Risk
    def define_risk(row):
        if row['Recency'] > 90:
            return '🔴 High Churn Risk'
        elif row['Recency'] > 45:
            return '🟡 Medium Risk'
        else:
            return '🟢 Loyal'

    rfm['Risk Status'] = rfm.apply(define_risk, axis=1)
    return rfm.reset_index(), snapshot_date

# ========================================================
# 4. PDF GENERATOR
# ========================================================
def generate_pdf(df_risk):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    pdf.cell(200, 10, txt="RetainIQ - Churn Risk Report", ln=True, align='C')
    pdf.ln(10)
    
    # Table Header
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(50, 10, "Customer", 1)
    pdf.cell(30, 10, "Days Silent", 1)
    pdf.cell(40, 10, "Total Spend", 1)
    pdf.cell(50, 10, "Status", 1)
    pdf.ln()
    
    # Rows
    pdf.set_font("Arial", size=10)
    for index, row in df_risk.iterrows():
        status = row['Risk Status']
        # Filter for PDF to only show risk
        if "High" in status or "Medium" in status:
            pdf.cell(50, 10, str(row['customer']), 1)
            pdf.cell(30, 10, str(row['Recency']), 1)
            pdf.cell(40, 10, f"${row['Monetary']:.2f}", 1)
            pdf.cell(50, 10, status, 1)
            pdf.ln()
            
    return pdf.output(dest='S').encode('latin-1')

# ========================================================
# 5. UI LAYOUT
# ========================================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/8649/8649607.png", width=60)
st.sidebar.title("RetainIQ Pro")
st.sidebar.info("Upload raw sales data (Name, Date, Amount) to detect hidden churn risks.")

st.title("🧠 RetainIQ: Transactional Churn Engine")

tab1, tab2 = st.tabs(["📂 Sales Data Upload", "🤖 AI Strategy"])

with tab1:
    st.header("Analyze Customer Sales Data")
    
    uploaded_file = st.file_uploader("Upload Excel/CSV (Required Columns: CustomerName, OrderDate, SalesAmount)", type=['csv', 'xlsx'])
    
    # Demo Data Button
    if st.button("Use Demo Data (Your SalesData.csv)"):
        # Create a mock dataframe mimicking your file structure
        data = {
            'OrderDate': pd.date_range(start='2024-01-01', periods=20).tolist() + pd.date_range(start='2023-01-01', periods=5).tolist(),
            'CustomerName': ['Kabir', 'Meera', 'Pooja', 'Aarav', 'Rohan']*5,
            'SalesAmount': [1200, 3500, 400, 21000, 500]*5
        }
        df_upload = pd.DataFrame(data)
        st.session_state['df_upload'] = df_upload
        st.success("Loaded Demo Data!")

    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df_upload = pd.read_csv(uploaded_file)
            else:
                df_upload = pd.read_excel(uploaded_file)
            st.session_state['df_upload'] = df_upload
            st.success("File Uploaded Successfully")
        except:
            st.error("Error reading file")

    # If data exists, process it
    if 'df_upload' in st.session_state:
        df = st.session_state['df_upload']
        
        # Check columns
        required = ['CustomerName', 'OrderDate', 'SalesAmount']
        if not all(col in df.columns for col in required):
            st.warning(f"Note: Your file should have columns roughly named: {required}. I will try to map them.")
        
        # PROCESS DATA
        risk_profile, last_date = process_sales_data(df)
        
        # --- DASHBOARD ---
        m1, m2, m3 = st.columns(3)
        m1.metric("Analysis Date", last_date.strftime('%Y-%m-%d'))
        m2.metric("Total Customers", len(risk_profile))
        high_risk = len(risk_profile[risk_profile['Risk Status'].str.contains("High")])
        m3.metric("⚠ At Risk Customers", high_risk, delta="-Action Needed", delta_color="inverse")
        
        st.markdown("---")
        
        c1, c2 = st.columns([2, 1])
        
        with c1:
            st.subheader("📉 Churn Risk vs Spending Power")
            fig = px.scatter(risk_profile, x="Recency", y="Monetary", 
                             color="Risk Status", size="Monetary",
                             color_discrete_map={'🔴 High Churn Risk':'red', '🟡 Medium Risk':'orange', '🟢 Loyal':'green'},
                             hover_data=['customer'],
                             title="Who hasn't purchased recently? (Right Side = Danger Zone)")
            st.plotly_chart(fig, use_container_width=True)
            
        with c2:
            st.subheader("📋 At-Risk List")
            # Filter only risk
            bad_customers = risk_profile[risk_profile['Risk Status'].str.contains("High|Medium")].sort_values('Recency', ascending=False)
            st.dataframe(bad_customers[['customer', 'Recency', 'Risk Status']], hide_index=True)
            
            # PDF Download
            pdf_bytes = generate_pdf(risk_profile)
            st.download_button(
                "⬇️ Download Risk PDF Report",
                data=pdf_bytes,
                file_name="Churn_Risk_Report.pdf",
                mime="application/pdf"
            )

with tab2:
    st.header("🤖 AI Retention Consultant")
    
    if 'df_upload' not in st.session_state:
        st.info("Please upload data in the first tab.")
    else:
        # Get bad customers again
        risk_profile, _ = process_sales_data(st.session_state['df_upload'])
        bad_customers = risk_profile[risk_profile['Risk Status'].str.contains("High")].head(10)
        
        if bad_customers.empty:
            st.success("No High Risk customers found!")
        else:
            selected_cust = st.selectbox("Select At-Risk Customer:", bad_customers['customer'].unique())
            
            cust_stats = risk_profile[risk_profile['customer'] == selected_cust].iloc[0]
            
            st.markdown(f"""
            ### Customer Profile: {selected_cust}
            - **Days Since Last Buy:** {cust_stats['Recency']} days 🚨
            - **Total Lifetime Spend:** ${cust_stats['Monetary']:.2f}
            - **Total Orders:** {cust_stats['Frequency']}
            """)
            
            if st.button("✨ Generate Recovery Plan"):
                if not groq_client:
                    st.error("Configure Groq API Key in Secrets.")
                else:
                    with st.spinner("Analyzing purchase history..."):
                        prompt = f"""
                        You are a Customer Success Manager.
                        Customer '{selected_cust}' used to be active but hasn't bought anything in {cust_stats['Recency']} days.
                        Their total spend is ${cust_stats['Monetary']}.
                        
                        1. Diagnose: Why might a customer stop buying after {cust_stats['Frequency']} orders?
                        2. Offer: Suggest a specific "Come Back" offer.
                        3. Email: Write a short, personalized email to them.
                        """
                        
                        completion = groq_client.chat.completions.create(
                            model="llama3-8b-8192",
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.7
                        )
                        st.markdown(completion.choices[0].message.content)
