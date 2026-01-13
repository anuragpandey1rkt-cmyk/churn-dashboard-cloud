import streamlit as st
import pandas as pd
import psycopg2
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import plotly.express as px

# 1. PAGE CONFIG
st.set_page_config(page_title="Real-Time Churn AI", layout="wide")

# 2. DATABASE CONNECTION FUNCTION
def get_connection():
    return psycopg2.connect(
        host=st.secrets["DB_HOST"],
        database=st.secrets["DB_NAME"],
        user=st.secrets["DB_USER"],
        password=st.secrets["DB_PASS"]
    )

# 3. INITIALIZE DATABASE (Run once to create table & dummy data)
def init_db():
    try:
        conn = get_connection()
        cur = conn.cursor()
        
        # Create Table if not exists
        cur.execute("""
            CREATE TABLE IF NOT EXISTS telecom_churn (
                customer_id SERIAL PRIMARY KEY,
                tenure INT,
                monthly_charges FLOAT,
                contract VARCHAR(50),
                churn VARCHAR(10)
            );
        """)
        
        # Check if empty, if so, add 500 dummy rows
        cur.execute("SELECT COUNT(*) FROM telecom_churn")
        if cur.fetchone()[0] == 0:
            st.warning("⚠️ Database empty. Generating dummy data...")
            data = []
            for _ in range(500):
                tenure = random.randint(1, 72)
                contract = random.choice(['Month-to-month', 'One year', 'Two year'])
                # Logic: Fiber is expensive
                base = 30
                monthly = round(base + random.uniform(10, 80), 2)
                
                # Logic: Churn probability
                score = (100 - tenure) + (monthly * 0.5)
                if contract == 'Month-to-month': score += 20
                churn = 'Yes' if score > 100 else 'No'
                
                data.append((tenure, monthly, contract, churn))
            
            args_str = ','.join(cur.mogrify("(%s,%s,%s,%s)", x).decode('utf-8') for x in data)
            cur.execute("INSERT INTO telecom_churn (tenure, monthly_charges, contract, churn) VALUES " + args_str)
            conn.commit()
            st.success("✅ Database seeded with 500 rows!")
            
        cur.close()
        conn.close()
    except Exception as e:
        st.error(f"Database Error: {e}")

# Run initialization
init_db()

# 4. FETCH DATA & TRAIN MODEL
@st.cache_data(ttl=60) # Cache for 60 seconds
def load_data_and_train():
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM telecom_churn", conn)
    conn.close()
    
    # Preprocessing
    le = LabelEncoder()
    df['contract_code'] = le.fit_transform(df['contract'])
    df['churn_code'] = df['churn'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    # Train Model
    X = df[['tenure', 'monthly_charges', 'contract_code']]
    y = df['churn_code']
    model = RandomForestClassifier()
    model.fit(X, y)
    
    return df, model, le

try:
    df, model, le = load_data_and_train()
    
    # 5. DASHBOARD UI
    st.title("📡 AI Telecom Churn Dashboard (Live DB)")
    
    # KPI Cards
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", len(df))
    col2.metric("Churn Rate", f"{df['churn_code'].mean()*100:.1f}%")
    col3.metric("Avg Monthly Bill", f"${df['monthly_charges'].mean():.2f}")
    
    # Charts
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Churn by Contract Type")
        fig = px.histogram(df, x="contract", color="churn", barmode="group")
        st.plotly_chart(fig, use_container_width=True)
        
    with c2:
        st.subheader("Monthly Charges vs Tenure")
        fig2 = px.scatter(df, x="tenure", y="monthly_charges", color="churn")
        st.plotly_chart(fig2, use_container_width=True)

    # 6. AI PREDICTION TOOL
    st.markdown("---")
    st.subheader("🤖 Predict Customer Risk")
    
    p_tenure = st.slider("Tenure (Months)", 1, 72, 12)
    p_charges = st.number_input("Monthly Charges ($)", 20, 150, 70)
    p_contract = st.selectbox("Contract Type", le.classes_)
    
    if st.button("Analyze Risk"):
        contract_val = le.transform([p_contract])[0]
        prediction = model.predict([[p_tenure, p_charges, contract_val]])
        prob = model.predict_proba([[p_tenure, p_charges, contract_val]])[0][1]
        
        if prediction[0] == 1:
            st.error(f"⚠️ HIGH RISK! This customer is {prob*100:.1f}% likely to leave.")
        else:
            st.success(f"✅ SAFE. This customer is likely to stay ({prob*100:.1f}% risk).")

except Exception as e:
    st.info("Waiting for database connection... Check Streamlit Secrets.")
    st.error(e)
