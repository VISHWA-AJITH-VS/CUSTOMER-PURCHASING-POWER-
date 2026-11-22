"""Data + analytics logic for the Customer Behaviour app (no Streamlit code here)."""
import os, shutil
from datetime import date
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SEED = os.path.join(DATA_DIR, "seed_customers.csv")
CUSTOMERS = os.path.join(DATA_DIR, "customers.csv")
PURCHASES = os.path.join(DATA_DIR, "purchases.csv")

# category key -> (column in customers.csv, nice label)
CATEGORIES = {
    "cooking_groceries":     ("cooking_groceries_spend_inr", "Cooking groceries"),
    "non_cooking_groceries": ("non_cooking_groceries_spend_inr", "Non-cooking groceries"),
    "vegetables":            ("vegetables_spend_inr", "Vegetables"),
    "fruits":                ("fruits_spend_inr", "Fruits"),
    "dairy":                 ("dairy_spend_inr", "Dairy"),
    "snacks_beverages":      ("snacks_beverages_spend_inr", "Snacks & beverages"),
    "household_items":       ("household_items_spend_inr", "Household items"),
    "personal_care":         ("personal_care_spend_inr", "Personal care"),
}
CAT_KEYS = list(CATEGORIES)
CAT_LABELS = {k: v[1] for k, v in CATEGORIES.items()}
PURCHASE_COLS = ["purchase_id", "date", "customer_id"] + CAT_KEYS + ["total", "payment_method"]
FEATURES = ["monthly_amount_spent_inr", "visits_per_month", "average_bill_amount_inr"]


# ------------------------------------------------------------------ storage
def init_data():
    """First run: create customers.csv from the seed file and an empty purchases.csv."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(CUSTOMERS):
        shutil.copy(SEED, CUSTOMERS)
    if not os.path.exists(PURCHASES):
        pd.DataFrame(columns=PURCHASE_COLS).to_csv(PURCHASES, index=False)


def load_customers():
    init_data()
    return pd.read_csv(CUSTOMERS, dtype={"phone_number": str})


def load_purchases():
    init_data()
    p = pd.read_csv(PURCHASES)
    p["date"] = pd.to_datetime(p["date"])
    return p


def next_customer_id(customers):
    nums = customers["customer_id"].str.extract(r"(\d+)")[0].astype(int)
    return f"CUST{(nums.max() if len(nums) else 0) + 1:04d}"


def add_customer(name, phone, annual_income, payment_method):
    c = load_customers()
    cid = next_customer_id(c)
    row = {col: 0 for col in c.columns}
    row.update(customer_id=cid, customer_name=name.strip(), phone_number=str(phone).strip(),
               annual_income_inr=int(annual_income), purchase_frequency="New",
               preferred_payment_method=payment_method, customer_segment_seed="New customer")
    c = pd.concat([c, pd.DataFrame([row])], ignore_index=True)
    c.to_csv(CUSTOMERS, index=False)
    return cid


def add_purchase(customer_id, purchase_date, amounts, payment_method):
    """Log a purchase, then refresh that customer's row in customers.csv."""
    p = load_purchases()
    pid = f"P{len(p) + 1:06d}"
    row = {"purchase_id": pid, "date": pd.Timestamp(purchase_date), "customer_id": customer_id,
           "total": int(sum(amounts.values())), "payment_method": payment_method}
    row.update({k: int(amounts.get(k, 0)) for k in CAT_KEYS})
    p = pd.concat([p, pd.DataFrame([row])], ignore_index=True)
    p.to_csv(PURCHASES, index=False)
    refresh_customer_rows([customer_id])
    return pid


# --------------------------------------------------------------- profiles
def profile_table(customers, purchases, window_days=30, today=None):
    """One row per customer with monthly-equivalent behaviour.

    If the customer has purchases logged in the last `window_days` days, the numbers come
    from the log (scaled to 30 days). Otherwise the baseline numbers from the dataset are used.
    """
    today = pd.Timestamp(today or date.today())
    prof = customers.copy()
    prof["profile_source"] = "baseline dataset"
    recent = purchases[(purchases["date"] > today - pd.Timedelta(days=window_days)) &
                       (purchases["date"] <= today)]
    if recent.empty:
        return prof
    scale = 30.0 / window_days
    g = recent.groupby("customer_id")
    sums = g[CAT_KEYS].sum() * scale
    visits = g.size() * scale
    bills = g["total"].mean()
    idx = prof["customer_id"].isin(sums.index)
    for k, (col, _) in CATEGORIES.items():
        prof.loc[idx, col] = prof.loc[idx, "customer_id"].map(sums[k]).round().astype(int)
    prof.loc[idx, "monthly_amount_spent_inr"] = prof.loc[idx, "customer_id"].map(sums.sum(axis=1)).round().astype(int)
    prof.loc[idx, "visits_per_month"] = prof.loc[idx, "customer_id"].map(visits).round(1)
    prof.loc[idx, "average_bill_amount_inr"] = prof.loc[idx, "customer_id"].map(bills).round().astype(int)
    prof.loc[idx, "profile_source"] = f"purchase log (last {window_days} days)"
    return prof


def refresh_customer_rows(customer_ids, window_days=30):
    """Write the up-to-date monthly profile of these customers back into customers.csv."""
    c = load_customers()
    prof = profile_table(c, load_purchases(), window_days)
    cols = [v[0] for v in CATEGORIES.values()] + ["monthly_amount_spent_inr", "visits_per_month",
                                                  "average_bill_amount_inr"]
    m = c["customer_id"].isin(customer_ids)
    for col in cols:
        c[col] = c[col].astype(float) if col == "visits_per_month" else c[col]
        c.loc[m, col] = prof.loc[m, col].values
    c.to_csv(CUSTOMERS, index=False)


# ------------------------------------------------------------ segmentation
def _name_segments(centres):
    med = centres.median()
    names = {}
    for cl, r in centres.iterrows():
        v, b, m = (r.visits_per_month / max(med.visits_per_month, 1e-9),
                   r.average_bill_amount_inr / max(med.average_bill_amount_inr, 1e-9),
                   r.monthly_amount_spent_inr / max(med.monthly_amount_spent_inr, 1e-9))
        if v >= 1.8 and b < 0.7:   n = "Small frequent shoppers"
        elif b >= 1.6 and v <= 0.6: n = "Premium occasional buyers"
        elif m >= 1.4:              n = "Family bulk buyers"
        elif m <= 0.75:             n = "Budget shoppers"
        else:                       n = "Balanced shoppers"
        names[cl] = n
    seen = {}
    for cl in names:                      # keep names unique
        seen[names[cl]] = seen.get(names[cl], 0) + 1
        if seen[names[cl]] > 1:
            names[cl] += f" ({seen[names[cl]]})"
    return names


def run_segmentation(profile, k=None):
    """K-Means on purchasing behaviour. Returns (profile with segment column, k, silhouette)."""
    prof = profile.copy()
    prof["segment"] = "Not enough data"
    ok = prof["monthly_amount_spent_inr"] > 0
    data = prof.loc[ok, FEATURES]
    if len(data) < 8:
        return prof, 0, float("nan")
    X = StandardScaler().fit_transform(data)
    if k is None:
        scores = {kk: silhouette_score(X, KMeans(kk, n_init=10, random_state=42).fit_predict(X))
                  for kk in range(3, min(7, len(data) - 1))}
        k = max(scores, key=scores.get)
    km = KMeans(k, n_init=20, random_state=42).fit(X)
    sil = silhouette_score(X, km.labels_)
    centres = data.assign(cl=km.labels_).groupby("cl")[FEATURES].mean()
    names = _name_segments(centres)
    prof.loc[ok, "segment"] = pd.Series(km.labels_, index=data.index).map(names)
    return prof, k, sil


# ------------------------------------------------------------------ insights
def income_band(income):
    if income < 150_000:   return "Lower income"
    if income < 500_000:   return "Middle income"
    return "Higher income"


def customer_insight(customer_id, purchases, start, end):
    p = purchases[(purchases["customer_id"] == customer_id) &
                  (purchases["date"] >= pd.Timestamp(start)) &
                  (purchases["date"] <= pd.Timestamp(end))].sort_values("date")
    out = {"purchases": p, "n": len(p)}
    if p.empty:
        return out
    cat_tot = p[CAT_KEYS].sum()
    out["cat_totals"] = cat_tot.rename(CAT_LABELS)
    out["cat_share"] = (cat_tot / cat_tot.sum() * 100).round(1).rename(CAT_LABELS)
    out["top_category"] = CAT_LABELS[cat_tot.idxmax()]
    out["total_spent"] = int(p["total"].sum())
    out["avg_bill"] = int(p["total"].mean())
    days = p["date"].dt.normalize().drop_duplicates().sort_values()
    out["avg_gap_days"] = float(days.diff().dt.days.mean()) if len(days) > 1 else None
    out["days_since_last"] = (pd.Timestamp(date.today()) - p["date"].max()).days
    out["fav_payment"] = p["payment_method"].mode().iat[0]
    out["fav_weekday"] = p["date"].dt.day_name().mode().iat[0]
    out["monthly_trend"] = p.set_index("date")["total"].resample("MS").sum()
    return out


def marketing_tips(segment, ins, spend_to_income):
    tips = []
    s = segment.lower()
    if "frequent" in s:  tips.append("Visits often with small bills: offer a loyalty-points card or 'add one more item' combos.")
    if "premium" in s:   tips.append("Rare but large bills: send premium-product launches and festival gift hampers.")
    if "family" in s:    tips.append("High monthly spend: offer bulk-pack and monthly-basket discounts.")
    if "budget" in s:    tips.append("Price-sensitive: promote weekly deals, coupons and value packs.")
    if "balanced" in s:  tips.append("Steady mid spender: cross-sell categories they buy least to grow basket size.")
    if ins.get("n"):
        tips.append(f"Top category is {ins['top_category']}: send targeted offers in it.")
        weakest = ins["cat_totals"].idxmin()
        tips.append(f"Spends least on {weakest}: a small intro discount could grow that category.")
        if ins.get("avg_gap_days"):
            tips.append(f"Buys about every {ins['avg_gap_days']:.0f} days: send a reminder just before that gap ends.")
        if ins["days_since_last"] > 2 * (ins.get("avg_gap_days") or 30):
            tips.append("Has not bought for a while: send a win-back offer.")
    if spend_to_income and spend_to_income > 100:
        tips.append("Recorded spending is above the stored income: check that income is up to date.")
    return tips
