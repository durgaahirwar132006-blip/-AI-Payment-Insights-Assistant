import os
from datetime import datetime, timedelta
import requests
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sklearn.ensemble import IsolationForest

load_dotenv()

st.set_page_config(page_title="AI Payment Insights Assistant", page_icon="💳", layout="wide")

st.title("💳 AI Payment Insights Assistant")
st.caption("Understand payment activity, spot unusual transactions, and test PayPal payments safely.")

with st.sidebar:
    st.header("PayPal Sandbox")
    st.write("Use **sandbox** credentials only. This prototype does not process real payments.")
    client_id = st.text_input("Sandbox Client ID", value=os.getenv("PAYPAL_CLIENT_ID", ""), type="password")
    client_secret = st.text_input("Sandbox Secret", value=os.getenv("PAYPAL_CLIENT_SECRET", ""), type="password")
    st.caption("Create these at developer.paypal.com → Apps & Credentials → Sandbox.")

def paypal_base():
    return "https://api-m.sandbox.paypal.com"

def get_paypal_token(client_id, client_secret):
    response = requests.post(
        f"{paypal_base()}/v1/oauth2/token",
        auth=(client_id, client_secret),
        headers={"Accept": "application/json", "Accept-Language": "en_US"},
        data={"grant_type": "client_credentials"},
        timeout=25,
    )
    response.raise_for_status()
    return response.json()["access_token"]

def make_sample_data():
    rng = np.random.default_rng(17)
    categories = ["Food", "Shopping", "Transport", "Bills", "Entertainment"]
    rows = []
    now = datetime.now()
    for i in range(90):
        day = now - timedelta(days=int(89-i))
        category = rng.choice(categories, p=[.30, .20, .18, .17, .15])
        typical = {"Food": 420, "Shopping": 1100, "Transport": 180, "Bills": 1500, "Entertainment": 650}[category]
        amount = max(30, float(rng.lognormal(np.log(typical), .38)))
        rows.append({
            "date": day.strftime("%Y-%m-%d"),
            "amount": round(amount, 2),
            "category": category,
            "description": f"{category} purchase",
        })
    # A few deliberately unusual sample values to make the demo understandable.
    rows[-4]["amount"] = 18500.00
    rows[-4]["description"] = "Unusually large purchase (synthetic demo)"
    rows[-13]["amount"] = 12400.00
    rows[-13]["description"] = "Large online purchase (synthetic demo)"
    return pd.DataFrame(rows)

def analyze_transactions(df):
    required = {"amount"}
    if not required.issubset(df.columns):
        raise ValueError("Your CSV must contain an 'amount' column.")
    clean = df.copy()
    clean["amount"] = pd.to_numeric(clean["amount"], errors="coerce")
    clean = clean.dropna(subset=["amount"])
    clean = clean[clean["amount"] >= 0].copy()
    if clean.empty:
        raise ValueError("No valid non-negative transaction amounts were found.")
    # The detector uses amount and log(amount); category is summarized separately.
    features = pd.DataFrame({
        "amount": clean["amount"].astype(float),
        "log_amount": np.log1p(clean["amount"].astype(float)),
    })
    if len(clean) >= 10:
        model = IsolationForest(n_estimators=150, contamination="auto", random_state=42)
        clean["anomaly_score"] = model.fit_predict(features)
        clean["unusual"] = clean["anomaly_score"] == -1
    else:
        clean["anomaly_score"] = 1
        clean["unusual"] = False
    return clean

def explain_flag(row, median):
    amount = float(row["amount"])
    if amount > median * 5:
        return f"₹{amount:,.2f} is more than 5× the dataset median. Check whether this payment is expected."
    if amount > median * 2:
        return f"₹{amount:,.2f} is over 2× the dataset median. It may be legitimate, but is worth reviewing."
    return f"₹{amount:,.2f} differs from the model's typical transaction pattern. Review the transaction details."

tab_overview, tab_transactions, tab_paypal, tab_about = st.tabs(
    ["📊 Insights", "🧾 Transactions", "🅿️ PayPal Sandbox", "ℹ️ About"]
)

with tab_overview:
    st.subheader("Explore transaction activity")
    uploaded = st.file_uploader("Upload a CSV (optional)", type=["csv"], help="Required column: amount. Optional: date, category, description.")
    try:
        raw = pd.read_csv(uploaded) if uploaded is not None else make_sample_data()
        if "category" not in raw.columns:
            raw["category"] = "Uncategorized"
        if "description" not in raw.columns:
            raw["description"] = "Transaction"
        analyzed = analyze_transactions(raw)
        median_amount = float(analyzed["amount"].median())
        total = float(analyzed["amount"].sum())
        unusual_count = int(analyzed["unusual"].sum())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Transactions analyzed", f"{len(analyzed):,}")
        c2.metric("Total transaction value", f"₹{total:,.2f}")
        c3.metric("Average payment", f"₹{analyzed['amount'].mean():,.2f}")
        c4.metric("Flagged for review", f"{unusual_count:,}")
        left, right = st.columns(2)
        with left:
            st.markdown("#### Transaction amounts")
            chart_df = analyzed.reset_index(drop=True).copy()
            chart_df.index = chart_df.index + 1
            st.line_chart(chart_df[["amount"]])
        with right:
            st.markdown("#### Total by category")
            category_totals = analyzed.groupby("category", dropna=False)["amount"].sum().sort_values(ascending=False)
            st.bar_chart(category_totals)
        st.markdown("#### AI-assisted review queue")
        flagged = analyzed[analyzed["unusual"]].copy()
        if flagged.empty:
            st.success("The model did not flag any transactions in this dataset. This does not prove every payment is safe.")
        else:
            flagged["review_reason"] = flagged.apply(lambda row: explain_flag(row, median_amount), axis=1)
            cols = [c for c in ["date", "description", "category", "amount", "review_reason"] if c in flagged.columns]
            st.dataframe(flagged[cols].sort_values("amount", ascending=False), use_container_width=True, hide_index=True)
        st.info("These are anomaly flags, not fraud determinations. The sample data is synthetic; model results depend on the uploaded dataset.")
    except Exception as exc:
        st.error(f"Could not analyze the file: {exc}")

with tab_transactions:
    st.subheader("Transaction table")
    st.write("Use a CSV with an `amount` column. Optional columns are `date`, `category`, and `description`.")
    st.code("date,amount,category,description\n2026-10-01,250,Food,Lunch\n2026-10-02,1800,Shopping,Shoes", language="csv")
    st.download_button(
        "Download example CSV",
        data=make_sample_data().to_csv(index=False).encode("utf-8"),
        file_name="sample_transactions.csv",
        mime="text/csv",
    )
    if uploaded is not None:
        st.dataframe(raw, use_container_width=True, hide_index=True)
    else:
        st.dataframe(make_sample_data(), use_container_width=True, hide_index=True)

with tab_paypal:
    st.subheader("PayPal Sandbox payment flow")
    st.warning("Sandbox only. Never paste live credentials here. Do not enter real customer or payment information.")
    st.markdown("This prototype can create a test order, show PayPal's approval link, and capture the order after approval.")
    if "paypal_order_id" not in st.session_state:
        st.session_state.paypal_order_id = ""
    amount = st.number_input("Test payment amount (USD)", min_value=1.00, max_value=100.00, value=5.00, step=1.00)
    if st.button("1. Connect to PayPal Sandbox"):
        if not client_id or not client_secret:
            st.error("Enter your PayPal Sandbox Client ID and Secret in the sidebar first.")
        else:
            try:
                st.session_state.paypal_token = get_paypal_token(client_id, client_secret)
                st.success("Connected to PayPal Sandbox.")
            except requests.RequestException as exc:
                st.error(f"PayPal authentication failed. Check sandbox credentials and internet access. Details: {exc}")
    if st.button("2. Create test order"):
        token = st.session_state.get("paypal_token")
        if not token:
            st.error("Connect to PayPal Sandbox first.")
        else:
            try:
                response = requests.post(
                    f"{paypal_base()}/v2/checkout/orders",
                    headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
                    json={
                        "intent": "CAPTURE",
                        "purchase_units": [{
                            "amount": {"currency_code": "USD", "value": f"{amount:.2f}"},
                            "description": "AI Payment Insights Assistant sandbox test",
                        }],
                        "application_context": {
                            "brand_name": "AI Payment Insights Assistant",
                            "user_action": "PAY_NOW",
                            "return_url": "https://example.com/return",
                            "cancel_url": "https://example.com/cancel",
                        },
                    },
                    timeout=25,
                )
                response.raise_for_status()
                order = response.json()
                st.session_state.paypal_order_id = order["id"]
                approval = next((item["href"] for item in order.get("links", []) if item.get("rel") == "approve"), None)
                st.success(f"Test order created: {order['id']}")
                if approval:
                    st.link_button("Open PayPal Sandbox approval page", approval)
                    st.caption("Sign in with a PayPal Sandbox personal/buyer account to approve the test payment.")
            except requests.RequestException as exc:
                st.error(f"Could not create the order: {exc}")
    order_id = st.text_input("3. Order ID to capture", value=st.session_state.paypal_order_id)
    if st.button("4. Capture approved test payment"):
        token = st.session_state.get("paypal_token")
        if not token or not order_id.strip():
            st.error("Connect to PayPal and provide an order ID first.")
        else:
            try:
                response = requests.post(
                    f"{paypal_base()}/v2/checkout/orders/{order_id.strip()}/capture",
                    headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
                    json={},
                    timeout=25,
                )
                if response.ok:
                    result = response.json()
                    st.success(f"Sandbox capture response: {result.get('status', 'received')}")
                    st.json(result)
                else:
                    st.error(f"Capture failed ({response.status_code}): {response.text}")
            except requests.RequestException as exc:
                st.error(f"Capture request failed: {exc}")

with tab_about:
    st.subheader("How this prototype works")
    st.markdown("""
- **PayPal:** uses the PayPal REST API sandbox to create and capture test orders.
- **Machine learning:** an Isolation Forest model flags transaction amounts that differ from the dataset's patterns.
- **Data analysis:** pandas and NumPy calculate totals and summaries; Streamlit presents the dashboard.
- **Privacy:** start with synthetic or anonymized CSV data. Do not upload sensitive financial information.
""")
    st.caption("Prototype limitation: the anomaly model uses transaction amounts only and is not a fraud-detection system. The natural-language review notes are transparent rule-based explanations, not output from a generative language model.")
