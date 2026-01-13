import streamlit as st
import pandas as pd
import psycopg2
import plotly.express as px
from groq import Groq
import io
from fpdf import FPDF

# ========================================================
# 1. FAIL-SAFE CONFIGURATION
# ========================================================
st.set_page_config(page_title="RetainIQ Pro", page_icon="🧠", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .metric-card {background-color: #f0f2f6; border-radius: 10px; padding: 15px; border-left: 5px solid #4B4BFF;}
    .risk-high {color: #FF4B4B; font-weight: bold; background-color: #ffe6e6; padding: 2px 5px; border-radius: 4px;}
    .risk-safe {color: #2E8B57; font-weight: bold; background-color: #e6ffe6; padding: 2px 5px; border-radius: 4px;}
    </style>
""", unsafe_allow_html=True)

# ========================================================
# 2. SYSTEM CONNECTIONS (WITH ERROR HANDLING)
# ========================================================

# Database Connection (Fail-Safe)
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

# AI Connection (Fail-Safe)
try:
    groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])
except:
    groq_client = None

# ========================================================
# 3. CORE LOGIC: SALES TO CHURN (RFM + PRODUCT)
# ========================================================
def analyze_sales_data(df):
    # 1. Clean Column Names (Remove spaces)
    df.columns = df.columns.str.strip()
    
    # 2. Auto-Map Your Specific Columns
    col_map = {
        'OrderDate': 'Date', 'date': 'Date',
        'CustomerName': 'Customer', 'Customer': 'Customer', 
        'SalesAmount': 'Amount', 'Amount': 'Amount', 'Sales': 'Amount',
        'Product': 'Item', 'Item': 'Item', 'ProductName': 'Item'
    }
    # Rename only if columns exist
    df = df.rename(columns=col_map)
    
    # 3. Data Type Cleaning
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(0)
    
    # Fill missing Item column if it doesn't exist
    if 'Item' not in df.columns:
        df['Item'] = 'Unknown Product'

    # 4. Reference Date (Simulation of "Today")
    snapshot_date = df['Date'].max()
    
    # 5. Aggregate Data per Customer (RFM)
    rfm = df.groupby('Customer').agg({
        'Date': lambda x: (snapshot_date - x.max()).days, # Recency
        'Customer': 'count', # Frequency
        'Amount': 'sum', # Monetary
        'Item': lambda x: x.mode()[0] if not x.mode().empty else "Unknown" # Favorite Item
    }).rename(columns={'Date': 'Days_Since_Last_Buy', 'Customer': 'Total_Orders', 'Amount': 'Total_Spent', 'Item': 'Favorite_Item'})
    
    # 6. Risk Logic
    def calculate_risk(row):
        if row['Days_Since_Last_Buy'] > 90:
            return 'High Risk'
        elif row['Days_Since_Last_Buy'] > 45:
            return 'Medium Risk'
        else:
            return 'Loyal'

    rfm['Status'] = rfm.apply(calculate_risk, axis=1)
    return rfm.reset_index(), snapshot_date

# ========================================================
# 4. PDF GENERATOR (Fixed for Rupees)
# ========================================================
def generate_pdf(df_risk):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "RetainIQ: Churn Risk Report", ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", 'B', 10)
    # Headers
    pdf.cell(50, 10, "Customer", 1)
    pdf.cell(30, 10, "Days Silent", 1)
    pdf.cell(40, 10, "Total Spent (Rs)", 1) # Using Rs to avoid PDF crashes
    pdf.cell(40, 10, "Top Item", 1)
    pdf.cell(30, 10, "Status", 1)
    pdf.ln()
    
    pdf.set_font("Arial", '', 9)
    # Data
    for _, row in df_risk.iterrows():
        if row['Status'] != "Loyal": # Only show risks
            pdf.cell(50, 10, str(row['Customer'])[:25], 1)
            pdf.cell(30, 10, str(row['Days_Since_Last_Buy']), 1)
            pdf.cell(40, 10, f"Rs. {row['Total_Spent']:.0f}", 1) 
            pdf.cell(40, 10, str(row['Favorite_Item'])[:20], 1)
            pdf.cell(30, 10, row['Status'], 1)
            pdf.ln()
            
    return pdf.output(dest='S').encode('latin-1')

# ========================================================
# 5. UI LAYOUT
# ========================================================
st.title("🧠 RetainIQ: Intelligent Customer Success")

# Sidebar Status
with st.sidebar:
    st.header("System Status")
    db_conn = get_db_connection()
    if db_conn:
        st.success("Database: Connected")
        db_conn.close()
    else:
        st.warning("Database: Disconnected (Using Upload Mode)")
        
    if groq_client:
        st.success("AI Brain: Active")
    else:
        st.error("AI Brain: Inactive (Check Key)")

# TABS
tab_upload, tab_ai = st.tabs(["📂 Transaction Analysis", "🤖 AI Consultant"])

# --------------------------------------------------------
# TAB 1: UPLOAD & ANALYZE
# --------------------------------------------------------
with tab_upload:
    st.markdown("### 📊 Sales Data Analyzer")
    st.info("Upload your Excel/CSV file containing: `CustomerName`, `OrderDate`, `SalesAmount`, `Product`")
    
    uploaded_file = st.file_uploader("Upload Sales Data", type=['csv', 'xlsx'])
    
    if uploaded_file:
        try:
            # Load Data
            if uploaded_file.name.endswith('.csv'):
                df_raw = pd.read_csv(uploaded_file)
            else:
                df_raw = pd.read_excel(uploaded_file)
                
            st.success(f"Successfully loaded {len(df_raw)} transactions.")
            
            # Process Data
            risk_df, last_date = analyze_sales_data(df_raw)
            
            # --- DASHBOARD ---
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Analysis Date", last_date.strftime('%Y-%m-%d'))
            col2.metric("Total Customers", len(risk_df))
            
            high_risk_count = len(risk_df[risk_df['Status'] == 'High Risk'])
            col3.metric("🚨 High Risk", high_risk_count)
            
            avg_spend = risk_df['Total_Spent'].mean()
            col4.metric("Avg Spend", f"₹{avg_spend:,.0f}") # UPDATED TO RUPEE
            
            st.divider()
            
            # CHARTS
            c1, c2 = st.columns([2, 1])
            with c1:
                st.subheader("📉 Risk Map (Days Silent vs. Spend)")
                fig = px.scatter(risk_df, x="Days_Since_Last_Buy", y="Total_Spent", 
                                 color="Status", size="Total_Spent",
                                 hover_data=['Customer', 'Favorite_Item'],
                                 color_discrete_map={'High Risk':'red', 'Medium Risk':'orange', 'Loyal':'green'},
                                 labels={"Total_Spent": "Total Spent (₹)", "Days_Since_Last_Buy": "Days Since Last Order"})
                st.plotly_chart(fig, use_container_width=True)
                
            with c2:
                st.subheader("📋 High Risk List")
                # Format dataframe for display
                display_df = risk_df[risk_df['Status'] == 'High Risk'][['Customer', 'Favorite_Item', 'Total_Spent']].copy()
                display_df['Total_Spent'] = display_df['Total_Spent'].apply(lambda x: f"₹{x:,.0f}")
                st.dataframe(display_df, hide_index=True)
                
                # PDF Download
                pdf_data = generate_pdf(risk_df)
                st.download_button("⬇️ Download PDF Report", data=pdf_data, file_name="Churn_Report.pdf", mime="application/pdf")
                
            # Store for AI Tab
            st.session_state['risk_df'] = risk_df
            
        except Exception as e:
            st.error(f"Error processing file: {e}")
            st.write("Please check that your file has columns like 'CustomerName', 'OrderDate', 'SalesAmount'")

# --------------------------------------------------------
# TAB 2: AI CONSULTANT
# --------------------------------------------------------
with tab_ai:
    st.markdown("### 🤖 AI Retention Specialist")
    
    if 'risk_df' not in st.session_state:
        st.warning("Please upload a file in the first tab to activate the AI.")
    else:
        risk_df = st.session_state['risk_df']
        high_risk_customers = risk_df[risk_df['Status'].str.contains("Risk")]
        
        if high_risk_customers.empty:
            st.success("No at-risk customers found! Good job.")
        else:
            # Customer Selector
            selected_customer = st.selectbox("Select a Customer to Save:", high_risk_customers['Customer'].unique())
            
            # Get Data
            cust_data = high_risk_customers[high_risk_customers['Customer'] == selected_customer].iloc[0]
            
            # Display Profile
            st.info(f"""
            **Profile:** {selected_customer}
            - 🛑 **Status:** {cust_data['Status']}
            - 🗓️ **Days Since Last Buy:** {cust_data['Days_Since_Last_Buy']} days
            - 💰 **Total Value:** ₹{cust_data['Total_Spent']:,.2f}
            - 🛍️ **Favorite Product:** {cust_data['Favorite_Item']}
            """)
            
            if st.button("✨ Generate AI Recovery Plan"):
                if not groq_client:
                    st.error("AI Key is missing. Check Streamlit Secrets.")
                else:
                    with st.spinner("AI is analyzing purchase history..."):
                        prompt = f"""
                        You are a Senior Customer Success Manager.
                        Target Customer: {selected_customer}
                        
                        Data:
                        - They haven't bought anything in {cust_data['Days_Since_Last_Buy']} days.
                        - They usually buy '{cust_data['Favorite_Item']}'.
                        - Their total lifetime spend is ₹{cust_data['Total_Spent']}.
                        
                        Task:
                        1. **Diagnosis:** Why might they have stopped buying '{cust_data['Favorite_Item']}'? (Give 1 likely reason).
                        2. **Offer:** Create a specific discount or bundle offer involving '{cust_data['Favorite_Item']}' to win them back.
                        3. **Email:** Write a short, warm, 3-sentence email sending them this offer. Use Rupees (₹) for currency.
                        """
                        
                        completion = groq_client.chat.completions.create(
                            model="llama3-8b-8192",
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.7
                        )
                        
                        response = completion.choices[0].message.content
                        st.markdown(response)
