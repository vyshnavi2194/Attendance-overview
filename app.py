# app.py
import os
from flask import Flask, render_template, request, send_file
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from io import BytesIO
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ---------- Configuration / expected column names ----------
DATA_FILE = r"C:\Users\vyshnavi.mahankali\Downloads\attendance_app\attendance_app\data\attendance.xlsx"

CHART_DIR = "static/charts"
N_CLUSTERS = 4  # can be tuned
RANDOM_STATE = 42

# column names (adjust if your excel has slightly different names)
COL_FAKEID = "Fake ID"
COL_ACCOUNT = "Account code"
COL_IN = "Avg. In Time"
COL_OUT = "Avg. Out Time"
COL_OFFICE = "Avg. Office hrs"
COL_BAY = "Avg. Bay hrs"
COL_BREAK = "Avg. Break hrs"
COL_CAFE = "Avg. Cafeteria hrs"
COL_OOO = "Avg. OOO hrs"
COL_UNBILLED = "Unbilled"
COL_HALF = "Half-Day leave"
COL_FULL = "Full-Day leave"
COL_ONLINE = "Online Check-in"
COL_EXEMPT = "Excemptions"
COL_UNALLOC = "Unallocated"

# ---------- App setup ----------
app = Flask(__name__)

if not os.path.exists(CHART_DIR):
    os.makedirs(CHART_DIR)

# ---------- Utility functions for parsing ----------
def parse_time_or_duration(x):
    """
    Accepts values like '08:12:26' (time-of-day) or '12:32:03' as durations.
    Returns float hours where meaningful (time -> hour + minute/60).
    """
    if pd.isna(x):
        return np.nan
    try:
        # If it's a pandas Timestamp or datetime-like, extract hour + minute
        if isinstance(x, (pd.Timestamp, pd.DatetimeTZDtype, pd.DatetimeIndex.__class__)):
            return x.hour + x.minute / 60.0 + x.second / 3600.0
    except Exception:
        pass

    # Try parse as timedelta (duration)
    try:
        td = pd.to_timedelta(x)
        hours = td.total_seconds() / 3600.0
        return hours
    except Exception:
        pass

    # Try parse as time-of-day
    try:
        t = pd.to_datetime(x).time()
        return t.hour + t.minute / 60.0 + t.second / 3600.0
    except Exception:
        return np.nan

def safe_mean(series):
    return float(series.dropna().mean()) if len(series.dropna())>0 else np.nan

# ---------- Load & preprocess dataset ----------
def load_and_prep(path=DATA_FILE):
    try:
        df = pd.read_excel(path)
    except FileNotFoundError:
        print(f"Error: Data file not found at {path}. Please check DATA_FILE path.")
        return pd.DataFrame()
    
    df = df.copy()

    # Normalize column names trimming spaces
    df.columns = [c.strip() for c in df.columns]

    # Parse numeric/time columns into float hours
    for col in [COL_IN, COL_OUT]:
        if col in df.columns:
            df[col + "_hr"] = df[col].apply(parse_time_or_duration)

    for col in [COL_OFFICE, COL_BAY, COL_BREAK, COL_CAFE, COL_OOO]:
        if col in df.columns:
            df[col + "_hrs_float"] = df[col].apply(parse_time_or_duration)

    # Numeric leave/checkin features
    for col in [COL_HALF, COL_FULL, COL_ONLINE]:
        if col in df.columns:
            df[col + "_num"] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Binarize some columns (Unallocated, Exemptions) if present
    if COL_UNALLOC in df.columns:
        df["unallocated_flag"] = df[COL_UNALLOC].astype(str).str.lower().isin(["yes", "true", "1"]).astype(int)
    else:
        df["unallocated_flag"] = 0

    if COL_EXEMPT in df.columns:
        df["exempt_flag"] = df[COL_EXEMPT].astype(str).str.lower().isin(["yes", "true", "1"]).astype(int)
    else:
        df["exempt_flag"] = 0

    # Derived features
    # Break/Bay ratio and OOO/Bay ratio
    df["break_bay_ratio"] = df.apply(lambda r: (r.get("Avg. Break hrs_hrs_float", np.nan) / max(0.001, r.get("Avg. Bay hrs_hrs_float", np.nan))) if not pd.isna(r.get("Avg. Break hrs_hrs_float", np.nan)) else np.nan, axis=1)
    df["ooo_bay_ratio"] = df.apply(lambda r: (r.get("Avg. OOO hrs_hrs_float", np.nan) / max(0.001, r.get("Avg. Bay hrs_hrs_float", np.nan))) if not pd.isna(r.get("Avg. OOO hrs_hrs_float", np.nan)) else np.nan, axis=1)

    # Fill NAs with medians for modeling
    numeric_cols = [c for c in df.columns if c.endswith("_hrs_float") or c.endswith("_hr") or c.endswith("_num") or c in ["unallocated_flag", "exempt_flag", "break_bay_ratio", "ooo_bay_ratio"]]
    for c in numeric_cols:
        if c in df.columns:
            median = df[c].median(skipna=True)
            df[c] = df[c].fillna(median)

    return df

# Load data
df = load_and_prep()

# Check if data loaded successfully before proceeding with modeling
if df.empty or len(df) < N_CLUSTERS:
    print("Insufficient data loaded. Skipping modeling.")
    FEATURES = []
    cluster_profiles = pd.DataFrame()
    kmeans = None
    iso = None
    df["cluster"] = np.nan
    df["anomaly_flag"] = 0
else:
    # ---------- Features for modeling ----------
    FEATURES = []
    for c in ["Avg. Bay hrs_hrs_float", "Avg. Break hrs_hrs_float", "Avg. OOO hrs_hrs_float", "Avg. Cafeteria hrs_hrs_float",
              "Avg. Office hrs_hrs_float", "break_bay_ratio", "ooo_bay_ratio", "Half-Day leave_num", "Full-Day leave_num"]:
        if c in df.columns:
            FEATURES.append(c)

    # Safety: if not enough features, fallback to some alternatives
    if len(FEATURES) < 3:
        FEATURES = [c for c in df.columns if ("_hrs_float" in c or "_hr" in c or "_num" in c)][:6]
    
    if not FEATURES:
        print("Error: No suitable features found for modeling.")
        FEATURES = []
        cluster_profiles = pd.DataFrame()
        kmeans = None
        iso = None
        df["cluster"] = np.nan
        df["anomaly_flag"] = 0
    else:
        # Standardize
        scaler = StandardScaler()
        X = scaler.fit_transform(df[FEATURES].values)

        # ---------- Clustering model ----------
        kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init='auto')
        kmeans.fit(X)
        df["cluster"] = kmeans.predict(X)

        # Cluster profiles (centers in original feature space)
        cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
        cluster_profiles = pd.DataFrame(cluster_centers, columns=FEATURES)
        cluster_profiles.index.name = "cluster"

        # ---------- Anomaly detector ----------
        iso = IsolationForest(contamination=0.05, random_state=RANDOM_STATE)
        iso.fit(X)
        df["anomaly_score"] = iso.decision_function(X)
        df["anomaly"] = iso.predict(X)
        df["anomaly_flag"] = df["anomaly"].apply(lambda v: 1 if v == -1 else 0)


# Helper to get row by Fake ID
def get_employee_row(emp_id):
    if df.empty:
        return None
    try:
        emp_id_int = int(emp_id)
    except ValueError:
        return None

    df_ids = pd.to_numeric(df[COL_FAKEID], errors="coerce")

    subset = df[df_ids == emp_id_int]
    if subset.empty:
        return None
    return subset.iloc[0]


# ---------- Charting helpers ----------
def save_bar_chart(values, labels, title, filename):
    plt.figure(figsize=(6,4))
    sns.barplot(x=labels, y=values, palette="rocket")
    plt.title(title, fontsize=12)
    plt.ylabel("Hours / Score", fontsize=10)
    plt.xlabel("")
    plt.xticks(rotation=15, ha="right", fontsize=9)
    plt.tight_layout()
    path = os.path.join(CHART_DIR, filename)
    plt.savefig(path)
    plt.close()
    return path

def save_account_distribution_chart(account_df, metric_col, title, filename):
    plt.figure(figsize=(6,4))
    sns.histplot(account_df[metric_col].dropna(), kde=False, bins=12, color='#0A3D62')
    plt.title(title, fontsize=12)
    plt.xlabel(metric_col.replace("_hrs_float", " (Hours)").replace("Avg. ", ""), fontsize=10)
    plt.ylabel("Employee Count", fontsize=10)
    plt.tight_layout()
    path = os.path.join(CHART_DIR, filename)
    plt.savefig(path)
    plt.close()
    return path

def save_cluster_distribution_chart(df_data, filename):
    plt.figure(figsize=(8, 4))
    cluster_counts = df_data['cluster'].value_counts().sort_index()
    # Using a mu-sigma inspired blue palette
    sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette=['#0A3D62', '#1B65A3', '#3A96D2', '#6CBEDF'])
    plt.title('Company-Wide Cluster Distribution', fontsize=14, color='#0A3D62')
    plt.xlabel('Cluster ID', fontsize=12)
    plt.ylabel('Number of Employees', fontsize=12)
    plt.xticks(fontsize=10)
    plt.tight_layout()
    path = os.path.join(CHART_DIR, filename)
    plt.savefig(path)
    plt.close()
    return path

# ---------- Story generation helpers ----------
def generate_employee_story(emp_row):
    if kmeans is None:
        return ["Model not loaded due to data error."], "Review data file path."
        
    emp_id = emp_row[COL_FAKEID]
    cluster = int(emp_row["cluster"])
    anom = bool(emp_row["anomaly_flag"] == 1)

    bay = emp_row.get("Avg. Bay hrs_hrs_float", np.nan)
    brk = emp_row.get("Avg. Break hrs_hrs_float", np.nan)
    ooo = emp_row.get("Avg. OOO hrs_hrs_float", np.nan)
    cafe = emp_row.get("Avg. Cafeteria hrs_hrs_float", np.nan)
    office = emp_row.get("Avg. Office hrs_hrs_float", np.nan)
    half = emp_row.get("Half-Day leave_num", 0)
    full = emp_row.get("Full-Day leave_num", 0)
    online_checkin = emp_row.get("Online Check-in_num", 0)

    z = {}
    for col in ["Avg. Bay hrs_hrs_float", "Avg. Break hrs_hrs_float", "Avg. OOO hrs_hrs_float", "Avg. Cafeteria hrs_hrs_float", "Avg. Office hrs_hrs_float"]:
        if col in df.columns:
            mu = df[col].mean()
            sigma = df[col].std(ddof=0) if df[col].std(ddof=0) != 0 else 1.0
            z[col] = (emp_row.get(col, mu) - mu) / sigma

    lines = []
    lines.append(f"Employee {emp_id} belongs to cluster #{cluster} (cluster profile reflects their peer group).")
    
    if z.get("Avg. Break hrs_hrs_float", 0) > 1.5:
        lines.append(f"🔴 **High Break Time:** Break time is significantly higher than company average ({brk:.1f} hrs), suggesting potential focus or time management issues.")
    elif z.get("Avg. Break hrs_hrs_float", 0) < -1:
        lines.append(f"🟢 **Low Break Time:** Break time is significantly lower than company average ({brk:.1f} hrs), potentially indicating burnout risk or non-reporting of breaks.")

    if z.get("Avg. Bay hrs_hrs_float", 0) < -1.5:
        lines.append(f"🔴 **Low Focused Hours:** Focused workstation hours (Bay hrs) are well below average ({bay:.1f} hrs), impacting focused delivery time.")
    elif z.get("Avg. Bay hrs_hrs_float", 0) > 1:
        lines.append(f"🟢 **High Focused Hours:** Bay hours are significantly above average ({bay:.1f} hrs), indicating strong desk presence and focus.")

    if z.get("Avg. OOO hrs_hrs_float", 0) > 1.5:
        lines.append(f"🟠 **High OOO Hours:** Out-of-office hours are substantially higher than peers ({ooo:.1f} hrs), potentially due to excessive meetings or client travel.")

    if office > 10.5 and bay < 7:
        lines.append(f"🟠 **Long Total Hours, Low Focus:** Total office time is long ({office:.1f} hrs), but focused Bay time is low. This suggests inefficiency or high non-Bay activity.")
        
    if online_checkin > 5:
         lines.append(f"🔵 **Frequent Remote Work:** Has logged {int(online_checkin)} online check-ins. Review if remote work is impacting team collaboration.")

    if half + full > 10:
        lines.append(f"🟠 **Frequent Leaves:** {int(half + full)} total leaves observed. Review personal factors or workload.")

    if anom:
        lines.append("🚨 **Model Anomaly:** This employee is flagged as an anomaly by the Isolation Forest model (behavior deviates significantly from peers).")

    if len(lines) == 1:
        lines.append("✅ No strong deviations detected; behavior is within expected bounds for the cluster.")

    rec = "Continue monitoring. This employee’s pattern is typical for their peer cluster."
    center = cluster_profiles.loc[cluster].to_dict() if cluster in cluster_profiles.index else {}
    
    if anom or z.get("Avg. Bay hrs_hrs_float", 0) < -1.5:
        rec = "🛑 **Immediate Action: Behavioral Review.** Schedule a one-on-one with the employee and manager to understand the root cause of the anomaly/low Bay hours and align expectations."
    elif z.get("Avg. Break hrs_hrs_float", 0) > 1.5:
        rec = "⚠️ **Time Management Coaching.** Recommend a specific coaching session on time-boxing, reducing unscheduled breaks, and using focused work techniques (e.g., Pomodoro)."
    elif (office > 11 and bay < 7):
        rec = "📊 **Workload & Efficiency Review.** Review the employee's project portfolio and work processes. The goal is to maximize 'Bay' time (focused work) and reduce non-value-add activities."
    elif z.get("Avg. OOO hrs_hrs_float", 0) > 1.5:
        rec = "🗓️ **Meeting Discipline.** Introduce guidelines for 'Bay Time' blocks, encourage rejection of non-essential meetings, and promote asynchronous communication for status updates."
    elif not anom and (half + full) < 3 and z.get("Avg. Bay hrs_hrs_float", 0) > 1.5:
        rec = "🌟 **Positive Reinforcement.** Acknowledge and recognize the employee's high focused-work discipline and commitment."

    return lines, rec

def generate_account_story(account_code):
    if df.empty:
        return ["Model not loaded due to data error."], "Review data file path."

    account_df = df[df[COL_ACCOUNT] == account_code]
    if account_df.empty:
        return ["No data for account."], "No recommendation."

    avg_bay = safe_mean(account_df.get("Avg. Bay hrs_hrs_float", pd.Series()))
    avg_break = safe_mean(account_df.get("Avg. Break hrs_hrs_float", pd.Series()))
    avg_ooo = safe_mean(account_df.get("Avg. OOO hrs_hrs_float", pd.Series()))
    unbilled_count = len(account_df[account_df.get(COL_UNBILLED, pd.Series()).astype(str).str.lower().str.contains("unbilled", na=False)])
    unbilled_pct = unbilled_count / max(1, len(account_df)) * 100

    cluster_counts = account_df["cluster"].value_counts(normalize=True).round(3).to_dict()
    anomaly_rate = account_df["anomaly_flag"].mean() * 100
    
    company_avg_bay = df["Avg. Bay hrs_hrs_float"].mean()
    company_avg_break = df["Avg. Break hrs_hrs_float"].mean()
    company_std_break = df["Avg. Break hrs_hrs_float"].std()

    lines = []
    lines.append(f"Account **{account_code}** has **{len(account_df)}** employees.")
    lines.append(f"Cluster distribution (normalized): {cluster_counts}.")
    lines.append(f"Avg Focused Bay hrs: **{avg_bay:.2f} hrs** (Company avg: {company_avg_bay:.2f} hrs).")
    lines.append(f"Avg Break hrs: **{avg_break:.2f} hrs** (Company avg: {company_avg_break:.2f} hrs).")
    lines.append(f"Avg OOO hrs: **{avg_ooo:.2f} hrs**.")
    lines.append(f"🔴 Unbilled proportion: **{unbilled_pct:.1f}%** (Total unbilled employees: {unbilled_count}).")
    lines.append(f"🚨 Model-flagged anomaly rate: **{anomaly_rate:.1f}%**.")

    rec = "✅ **Standard Performance.** Continue with regular performance management cycles."
    
    if avg_break > company_avg_break + 1.5 * company_std_break:
        rec = "⚠️ **Account-Wide Discipline Review.** The account shows significantly higher break time. Recommend management intervention to enforce time-discipline and structured breaks across the team."
    if avg_bay < company_avg_bay - 1.5:
        rec = "📉 **Productivity Workshop.** Account shows low focused-work hours. Recommend a structured productivity/efficiency workshop focused on task prioritization and minimizing distractions."
    if unbilled_pct > 25:
        rec = "💰 **Urgent Billing/Allocation Review.** A large fraction of unbilled resources. Prioritize a review of project allocation and ensure all effort is mapped to billable or strategic work."
    if anomaly_rate > 10:
        rec = "🚨 **Deep Dive on Exceptions.** High anomaly rate suggests systemic behavioral deviations. Run a deeper review of individual attendance records and exceptions within this account."
    if avg_bay > company_avg_bay + 1 and avg_break < company_avg_break - 0.5 and anomaly_rate < 5 and unbilled_pct < 10:
         rec = "🏆 **Account for Recognition.** High focused work hours and low red flags. Consider this account as a benchmark for best practices."

    return lines, rec

# ---------- Routes ----------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/overview")
def overview():
    if df.empty or 'cluster' not in df.columns or kmeans is None:
        return render_template("overview.html", error="Data or model not loaded properly. Check data file path.")

    total_employees = len(df)

    # 1. Key Metrics
    avg_bay = df["Avg. Bay hrs_hrs_float"].mean()
    avg_break = df["Avg. Break hrs_hrs_float"].mean()
    anomaly_rate = df["anomaly_flag"].mean() * 100
    unbilled_pct = (df[COL_UNBILLED].astype(str).str.lower().str.contains("unbilled", na=False).sum() / max(1, total_employees)) * 100

    # 2. Charts
    cluster_chart_path = save_cluster_distribution_chart(df, "company_cluster_dist.png")

    hist_fn = "company_office_hist.png"
    office_hist_path = save_account_distribution_chart(df, "Avg. Office hrs_hrs_float", "Company Office Hours Distribution", hist_fn)

    # Bay vs Break comparison
    plt.figure(figsize=(6,4))
    sns.boxplot(data=df[["Avg. Bay hrs_hrs_float","Avg. Break hrs_hrs_float"]], palette="Set2")
    plt.title("Bay vs Break Hours (Company-wide)", fontsize=13)
    plt.ylabel("Hours", fontsize=11)
    plt.xticks([0,1], ["Bay Hours","Break Hours"])
    plt.tight_layout()
    bay_break_path = os.path.join(CHART_DIR, "bay_break.png")
    plt.savefig(bay_break_path)
    plt.close()

    # 3. Anomaly Summary
    anomaly_summary = df[df["anomaly_flag"] == 1].sort_values(by="anomaly_score").head(5)[[COL_FAKEID, COL_ACCOUNT, "anomaly_score", "cluster"]].rename(columns={"anomaly_score": "Score"})

    # 4. Account-Level Summary (Top 5 by unbilled %)
    account_unbilled = df.groupby(COL_ACCOUNT).apply(
        lambda x: (x[COL_UNBILLED].astype(str).str.lower().str.contains("unbilled", na=False).sum() / max(1,len(x))) * 100
    ).reset_index(name="Unbilled %").sort_values(by="Unbilled %", ascending=False).head(5)

    # 5. Executive Recommendation
    recommendation = "✅ Performance is within expected norms."
    if anomaly_rate > 10:
        recommendation = "🚨 High anomaly rate (>10%). Recommend immediate deep-dive into behavioral deviations."
    elif avg_bay < 7:
        recommendation = "📉 Average Bay hours below 7. Suggest a company-wide productivity workshop."
    elif unbilled_pct > 20:
        recommendation = "💰 More than 20% employees are unbilled. Urgent allocation and billing review required."
    elif avg_bay > 8 and avg_break < 1 and anomaly_rate < 5 and unbilled_pct < 10:
        recommendation = "🏆 Strong performance. Company is well above norms in productivity with minimal anomalies."

    return render_template("overview.html",
                           total_employees=total_employees,
                           avg_bay=f"{avg_bay:.2f}",
                           avg_break=f"{avg_break:.2f}",
                           anomaly_rate=f"{anomaly_rate:.1f}",
                           unbilled_pct=f"{unbilled_pct:.1f}",
                           cluster_chart=cluster_chart_path,
                           office_hist=office_hist_path,
                           bay_break=bay_break_path,
                           anomaly_table=anomaly_summary.to_html(index=False, classes='data-table'),
                           account_table=account_unbilled.to_html(index=False, classes='data-table'),
                           recommendation=recommendation)


@app.route("/employee", methods=["GET", "POST"])
def employee():
    if request.method == "POST":
        emp_id = request.form.get("emp_id", "").strip()
        if emp_id == "":
            return render_template("employee.html", error="Please enter an Employee ID.")

        row = get_employee_row(emp_id)
        if row is None:
            return render_template("employee.html", emp_id=emp_id, error="Employee ID not found in dataset or dataset is empty.")

        labels = ["Bay Hrs", "Break Hrs", "OOO Hrs", "Cafeteria", "Office Hrs"]
        values = [
            row.get("Avg. Bay hrs_hrs_float", 0),
            row.get("Avg. Break hrs_hrs_float", 0),
            row.get("Avg. OOO hrs_hrs_float", 0),
            row.get("Avg. Cafeteria hrs_hrs_float", 0),
            row.get("Avg. Office hrs_hrs_float", 0)
        ]
        chart_fn = f"{emp_id}_employee_bar.png"
        chart_path = save_bar_chart(values, labels, f"Work Pattern: {emp_id}", chart_fn)

        story_lines, recommendation = generate_employee_story(row)

        hist_fn = f"{emp_id}_bay_hist.png"
        plt.figure(figsize=(6,4))
        sns.histplot(df["Avg. Bay hrs_hrs_float"].dropna(), bins=15, kde=False, color='#0A3D62')
        plt.axvline(x=row.get("Avg. Bay hrs_hrs_float", 0), color="#FF884B", linestyle="--", linewidth=3, label=f"{emp_id} Bay hrs")
        plt.legend()
        plt.title("Company Bay Hours Distribution", fontsize=12)
        plt.xlabel("Avg. Bay hrs (Hours)", fontsize=10)
        plt.ylabel("Employee Count", fontsize=10)
        plt.tight_layout()
        hist_path = os.path.join(CHART_DIR, hist_fn)
        plt.savefig(hist_path)
        plt.close()

        return render_template("employee.html",
                               emp_id=emp_id,
                               chart=chart_path,
                               hist=hist_path,
                               story=story_lines,
                               recommendation=recommendation)
    return render_template("employee.html")

@app.route("/account", methods=["GET", "POST"])
def account():
    if request.method == "POST":
        acc_code = request.form.get("acc_code", "").strip()
        if acc_code == "":
            return render_template("account.html", error="Please enter an account code.")
        account_df = df[df[COL_ACCOUNT] == acc_code]
        if account_df.empty:
            return render_template("account.html", acc_code=acc_code, error="Account not found or no employees for this account.")

        avg_bay = account_df["Avg. Bay hrs_hrs_float"].mean()
        avg_break = account_df["Avg. Break hrs_hrs_float"].mean()
        avg_ooo = account_df["Avg. OOO hrs_hrs_float"].mean()
        avg_cafe = account_df["Avg. Cafeteria hrs_hrs_float"].mean()
        labels = ["Avg Bay", "Avg Break", "Avg OOO", "Avg Cafe"]
        values = [avg_bay, avg_break, avg_ooo, avg_cafe]
        chart_fn = f"{acc_code}_account_bar.png"
        chart_path = save_bar_chart(values, labels, f"Account Work Pattern: {acc_code}", chart_fn)

        hist_fn = f"{acc_code}_office_hist.png"
        hist_path = save_account_distribution_chart(account_df, "Avg. Office hrs_hrs_float", "Account Office Hours Distribution", hist_fn)

        story_lines, recommendation = generate_account_story(acc_code)

        return render_template("account.html",
                               acc_code=acc_code,
                               chart=chart_path,
                               hist=hist_path,
                               story=story_lines,
                               recommendation=recommendation)
    return render_template("account.html")

# ---------- Run ----------
if __name__ == "__main__":
    print("Starting Attendance Insights App...")
    app.run(debug=True)