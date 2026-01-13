import streamlit as st
import pandas as pd
import psycopg2
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# 1. PAGE CONFIG
st.set_page_config(page_title="EcoWise AI Churn Dashboard", page_icon="📉", layout="wide")

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
        st.error("🚨 **Database Connection Failed!**")
        st.write("Please check your Streamlit Secrets.")
        st.code(str(e)) # Shows the exact error
        st.stop() # Stops the app so it doesn't crash later

# 3. INITIALIZE DB (Create Table if Missing)
def init_db():
    conn = get_connection()
    cur = conn.cursor()
    try:
        # Check if table exists
        cur.execute("""
            CREATE TABLE IF NOT EXISTS telecom_churn (
                customer_id SERIAL PRIMARY KEY,
                tenure INT,
                monthly_charges FLOAT,
                contract VARCHAR(50),
                churn VARCHAR(10)
            );
        """)
        
        # Check if empty
        cur.execute("SELECT COUNT(*) FROM telecom_churn")
        count = cur.fetchone()[0]
        if count == 0:
            st.warning("⚠️ Database was empty. Seeding dummy data...")
            # Insert Dummy Data
            data_values = []
            for i in range(100): # Create 100 rows
                data_values.append(f"({(i%72)+1}, {(i%100)+20}, 'Month-to-month', '{'Yes' if i%2==0 else 'No'}')")
            
            query = "INSERT INTO telecom_churn (tenure, monthly_charges, contract, churn) VALUES " + ",".join(data_values)
            cur.execute(query)
            conn.commit()
            st.success("✅ Database seeded successfully!")
            
    except Exception as e:
        st.error(f"Error initializing database: {e}")
    finally:
        cur.close()
        conn.close()

# Run Init (Only runs if needed)
init_db()

# 4. LOAD DATA & TRAIN MODEL (No Cache to prevent stuck errors)
def load_data():
    conn = get_connection()
    try:
        df = pd.read_sql("SELECT * FROM telecom_churn", conn)
        return df
    except Exception as e:
        st.error(f"Error reading data: {e}")
        st.stop()
    finally:
        conn.close()

# Fetch Data
df = load_data()

# SAFETY CHECK: If df is empty, stop here
if df is None or df.empty:
    st.warning("No data found in the database yet.")
    st.stop()

# 5. PREPROCESS & TRAIN
le = LabelEncoder()
df['contract_code'] = le.fit_transform(df['contract'])
df['churn_code'] = df['churn'].apply(lambda x: 1 if x == 'Yes' else 0)

X = df[['tenure', 'monthly_charges', 'contract_code']]
y = df['churn_code']
model = RandomForestClassifier()
model.fit(X, y)

# 6. DASHBOARD LAYOUT
st.title("📉 EcoWise AI: Customer Retention Dashboard")

# Metric Cards
col1, col2, col3 = st.columns(3)
col1.metric("Total Customers", len(df))
col2.metric("Churn Rate", f"{(df['churn_code'].mean() * 100):.1f}%")
col3.metric("Avg Monthly Bill", f"${df['monthly_charges'].mean():.2f}")

# Charts
c1, c2 = st.columns(2)
with c1:
    st.subheader("Churn by Contract")
    fig = px.histogram(df, x="contract", color="churn", barmode="group", color_discrete_map={'Yes':'red', 'No':'green'})
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.subheader("Charges vs Tenure")
    fig2 = px.scatter(df, x="tenure", y="monthly_charges", color="churn", color_discrete_map={'Yes':'red', 'No':'green'})
    st.plotly_chart(fig2, use_container_width=True)

# AI Prediction Section
st.markdown("---")
st.subheader("🤖 Live AI Churn Predictor")
st.write("Tweaking these values sends a live query to the ML model.")

c_input1, c_input2, c_input3 = st.columns(3)
p_tenure = c_input1.slider("Tenure (Months)", 1, 72, 12)
p_charges = c_input2.number_input("Monthly Charges ($)", 20, 200, 70)
p_contract = c_input3.selectbox("Contract Type", le.classes_)

if st.button("Predict Risk"):
    # Encode Input
    p_contract_code = le.transform([p_contract])[0]
    
    # Predict
    prob = model.predict_proba([[p_tenure, p_charges, p_contract_code]])[0][1]
    
    if prob > 0.5:
        st.error(f"⚠️ **High Risk!** Probability of Churn: {prob*100:.1f}%")
    else:
        st.success(f"✅ **Safe Customer.** Probability of Churn: {prob*100:.1f}%")
