"""Customer Behaviour & Segmentation app.   Run:  streamlit run app.py"""
from datetime import date, timedelta
import pandas as pd
import streamlit as st
import logic as L

st.set_page_config(page_title="Customer Behaviour App", page_icon="🛒", layout="wide")
L.init_data()

page = st.sidebar.radio("Menu", ["📊 Dashboard", "➕ Add customer", "🧾 Record purchase",
                                 "🔍 Customer insights", "🧩 Segmentation"])
st.sidebar.caption("Data is saved in the data/ folder (customers.csv, purchases.csv).")

customers = L.load_customers()
purchases = L.load_purchases()


def label(row):
    return f"{row.customer_id} - {row.customer_name}"


def pick_customer(key):
    opts = customers.apply(label, axis=1).tolist()
    choice = st.selectbox("Customer (type to search)", opts, key=key)
    return choice.split(" - ")[0]


# ------------------------------------------------------------- Dashboard
if page == "📊 Dashboard":
    st.title("🛒 Customer Behaviour Dashboard")
    prof, k, sil = L.run_segmentation(L.profile_table(customers, purchases))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", len(customers))
    c2.metric("Purchases logged", len(purchases))
    c3.metric("Avg monthly spend (₹)", f"{prof['monthly_amount_spent_inr'].mean():,.0f}")
    c4.metric("Avg visits / month", f"{prof['visits_per_month'].mean():.1f}")

    st.subheader("Customers per segment")
    st.bar_chart(prof["segment"].value_counts())
    st.subheader("Average monthly spend by category (₹)")
    cat_cols = {v[0]: v[1] for v in L.CATEGORIES.values()}
    st.bar_chart(prof[list(cat_cols)].rename(columns=cat_cols).mean().round(0))
    if not purchases.empty:
        st.subheader("Recent purchases")
        st.dataframe(purchases.sort_values("date", ascending=False).head(10), use_container_width=True)

# ------------------------------------------------------- Add customer
elif page == "➕ Add customer":
    st.title("➕ Add a new customer")
    st.info(f"The next customer ID will be **{L.next_customer_id(customers)}**")
    with st.form("new_customer", clear_on_submit=True):
        name = st.text_input("Customer name")
        phone = st.text_input("Phone number")
        income = st.number_input("Annual income (₹)", min_value=0, value=120000, step=10000)
        pay = st.selectbox("Preferred payment method", ["UPI", "Card", "Cash"])
        ok = st.form_submit_button("Add customer")
    if ok:
        if not name.strip():
            st.error("Please enter the customer name.")
        elif phone.strip() and (not phone.strip().isdigit() or len(phone.strip()) < 10):
            st.error("Phone number should have at least 10 digits.")
        else:
            cid = L.add_customer(name, phone, income, pay)
            st.success(f"Added {name} with ID {cid}. Now record their purchases.")

# ----------------------------------------------------- Record purchase
elif page == "🧾 Record purchase":
    st.title("🧾 Record a purchase")
    cid = pick_customer("buy_cust")
    with st.form("purchase", clear_on_submit=True):
        d = st.date_input("Purchase date", value=date.today(), max_value=date.today())
        pay = st.selectbox("Payment method", ["UPI", "Card", "Cash"])
        st.write("Amount spent (₹) in each category:")
        cols = st.columns(2)
        amounts = {}
        for i, key in enumerate(L.CAT_KEYS):
            amounts[key] = cols[i % 2].number_input(L.CAT_LABELS[key], min_value=0, value=0, step=50, key=f"amt_{key}")
        ok = st.form_submit_button("Save purchase")
    if ok:
        total = sum(amounts.values())
        if total <= 0:
            st.error("Enter at least one amount greater than 0.")
        else:
            L.add_purchase(cid, d, amounts, pay)
            st.success(f"Saved purchase of ₹{total:,} for {cid}. The dataset has been updated.")
    mine = purchases[purchases["customer_id"] == cid].sort_values("date", ascending=False)
    st.subheader("This customer's logged purchases")
    if mine.empty:
        st.caption("No purchases logged yet.")
    else:
        st.dataframe(mine.head(10), use_container_width=True)

# ------------------------------------------------- Customer insights
elif page == "🔍 Customer insights":
    st.title("🔍 Customer insights")
    cid = pick_customer("ins_cust")
    row = customers[customers["customer_id"] == cid].iloc[0]
    prof, k, _ = L.run_segmentation(L.profile_table(customers, purchases))
    me = prof[prof["customer_id"] == cid].iloc[0]

    st.subheader(f"{row.customer_name}  ({cid})")
    band = L.income_band(row.annual_income_inr)
    est_annual = me.monthly_amount_spent_inr * 12
    ratio = est_annual / row.annual_income_inr * 100 if row.annual_income_inr else 0
    a, b, c, d = st.columns(4)
    a.metric("Income status", band)
    b.metric("Annual income (₹)", f"{row.annual_income_inr:,}")
    c.metric("Segment", me.segment)
    d.metric("Spend vs income", f"{ratio:.0f}%")
    st.caption(f"Profile source: {me.profile_source}. Spend vs income = (monthly spend × 12) ÷ annual income.")

    st.markdown("#### Time interval")
    quick = st.radio("Period", ["Last 7 days", "Last 30 days", "Last 90 days", "Last 365 days", "Custom"],
                     horizontal=True, index=1)
    if quick == "Custom":
        r = st.date_input("Choose dates", value=(date.today() - timedelta(days=30), date.today()))
        start, end = (r if len(r) == 2 else (r[0], r[0]))
    else:
        start, end = date.today() - timedelta(days=int(quick.split()[1])), date.today()

    ins = L.customer_insight(cid, purchases, start, end)
    if ins["n"] == 0:
        st.warning("No purchases logged in this period. Record purchases to see behaviour here.")
        st.markdown("**Baseline monthly spend from the original dataset (₹)**")
        base = {v[1]: int(row[v[0]]) for v in L.CATEGORIES.values()}
        st.bar_chart(pd.Series(base))
    else:
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total spent (₹)", f"{ins['total_spent']:,}")
        m2.metric("Purchases", ins["n"])
        m3.metric("Avg bill (₹)", f"{ins['avg_bill']:,}")
        m4.metric("Avg days between visits", f"{ins['avg_gap_days']:.1f}" if ins["avg_gap_days"] else "n/a")
        m5.metric("Days since last visit", ins["days_since_last"])

        l, r_ = st.columns(2)
        with l:
            st.markdown("**Spending by category (₹)**")
            st.bar_chart(ins["cat_totals"])
        with r_:
            st.markdown("**Share of spending (%)**")
            st.dataframe(ins["cat_share"].sort_values(ascending=False).rename("share %"), use_container_width=True)
        st.markdown("**Spending over time (₹ per month)**")
        st.line_chart(ins["monthly_trend"])
        st.write(f"Most bought category: **{ins['top_category']}** · Usual payment: **{ins['fav_payment']}** · "
                 f"Usual day: **{ins['fav_weekday']}**")
        st.markdown("**Purchase history**")
        st.dataframe(ins["purchases"], use_container_width=True)

    st.markdown("#### 🎯 Marketing suggestions")
    for t in L.marketing_tips(me.segment, ins, ratio):
        st.write("• " + t)

# ----------------------------------------------------- Segmentation
else:
    st.title("🧩 Customer segmentation (K-Means)")
    window = st.slider("Use purchases from the last N days (customers without recent purchases use baseline data)",
                       7, 180, 30)
    auto = st.checkbox("Choose number of segments automatically", value=True)
    k = None if auto else st.slider("Number of segments (k)", 2, 8, 5)
    prof, k_used, sil = L.run_segmentation(L.profile_table(customers, purchases, window), k)
    if k_used == 0:
        st.warning("Not enough customers with spending data to cluster.")
    else:
        st.success(f"{k_used} segments · silhouette score {sil:.2f} (closer to 1 = better separated)")
        feats = L.FEATURES
        summary = prof[prof.segment != "Not enough data"].groupby("segment").agg(
            customers=("customer_id", "count"),
            avg_monthly_spend=("monthly_amount_spent_inr", "mean"),
            avg_visits=("visits_per_month", "mean"),
            avg_bill=("average_bill_amount_inr", "mean")).round(1)
        st.dataframe(summary, use_container_width=True)
        st.markdown("**Spending vs visits**")
        st.scatter_chart(prof[prof.segment != "Not enough data"], x="visits_per_month",
                         y="monthly_amount_spent_inr", color="segment")
        cat_cols = {v[0]: v[1] for v in L.CATEGORIES.values()}
        st.markdown("**What each segment buys (avg ₹/month)**")
        st.bar_chart(prof.groupby("segment")[list(cat_cols)].mean().rename(columns=cat_cols))
        st.download_button("⬇️ Download customers with segments",
                           prof.to_csv(index=False).encode(), "customers_with_segments.csv", "text/csv")
