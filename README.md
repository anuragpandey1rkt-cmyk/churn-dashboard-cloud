# 📉 RetainIQ – Customer Churn Analysis Dashboard

RetainIQ is a **Customer Churn Analysis Dashboard** built using **Streamlit** that identifies customers at risk of churn by analyzing **transactional behavior**.  
The system does **not rely on pre-labeled churn data**. Instead, it derives churn risk using **behavioral metrics** such as recency, frequency, and customer lifetime value (LTV).

---

## 🎯 Problem Statement

Customer churn is a critical challenge for businesses, leading to revenue loss and reduced customer lifetime value.  
Most organizations lack tools to **proactively identify churn risk** using real behavioral data.

**Objective:**  
To design an interactive dashboard that:
- Detects churn risk from transaction history
- Segments customers by lifecycle stage
- Supports data-driven retention decisions

---

## 🚀 Key Features

### 🔍 Behavioral Churn Detection
- Churn risk is **derived**, not pre-defined
- Based on:
  - Days since last purchase (Recency)
  - Purchase frequency
  - Customer lifetime value (LTV)

### 👥 Customer Lifecycle Segmentation
Customers are categorized into:
- ✨ New Customer
- 🟢 Active
- ⭐ Loyal
- ⚠️ At Risk
- 🔴 Churned (Highly Likely)

### 📊 Executive Dashboard
- Total Customers
- Churn Rate
- At-Risk Customers
- Revenue at Risk
- Monthly Active Customer Trend
- Churn Risk Matrix (Recency vs Frequency)

### 📋 Priority Action List
- Customers ranked by **Churn Risk Score (0–100)**
- Helps retention teams focus on high-impact users
- Exportable as Excel

### 🤖 AI Retention Consultant
- Uses LLMs to generate:
  - Churn diagnosis
  - Personalized retention offers
  - Email communication drafts

---

## 🧠 Churn Logic (Core Methodology)

The dashboard uses an **RFM-based behavioral model**:

- **Recency:** Days since last transaction  
- **Frequency:** Number of purchases  
- **Monetary (LTV):** Total spend  

### Churn Risk Score (0–100)
Churn Risk Score =
0.7 × Recency Score + 0.3 × Frequency Score

**Business Rules:**
- Recency > 90 days → 🔴 Churned
- Recency 45–90 days → ⚠️ At Risk
- No manual churn labels are used

---

## 📁 Input Data Format

Upload a CSV or Excel file with the following columns:

| Column Name | Description |
|------------|-------------|
| CustomerName | Customer identifier |
| OrderDate | Transaction date |
| SalesAmount | Transaction value |
| Product | Product purchased |

The dashboard automatically derives all churn metrics.

---

## 🛠️ Tech Stack

- **Frontend & App Framework:** Streamlit  
- **Data Processing:** Pandas  
- **Visualization:** Plotly  
- **AI Integration:** Groq LLM API  
- **Database (Optional):** PostgreSQL  
- **Deployment:** Streamlit Cloud  

---

## 🔐 Authentication

- Supports database-based login
- **Demo Access:**  
  - Username: `admin`  
  - Password: `admin`  
- Designed for evaluator-friendly access

---

## 🌐 Live Application

👉 **Live App:**  
https://churn-dashboard-cloud-2026.streamlit.app/

---

## 📌 Use Cases

- Customer Retention Strategy
- Revenue Risk Analysis
- Marketing Campaign Targeting
- Academic & Internship Projects
- AI-powered Business Dashboards

---

## 🏁 Conclusion

RetainIQ demonstrates how **transactional data** can be transformed into **actionable churn insights** without requiring labeled churn datasets.  
The project bridges **business intelligence, data science, and AI**, making it suitable for real-world retention analytics.

---

## 👤 Author

**Anurag Pandey**  
Customer Analytics & AI Projects  
