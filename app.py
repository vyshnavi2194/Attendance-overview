import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans 
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import math
import time
warnings.filterwarnings("ignore")

# Set Streamlit Page Configuration
st.set_page_config(layout="wide", page_title="Attendance Analytics Dashboard")

# ---------- MuSigma Brand Configuration ----------
MUSIGMA_PRIMARY = "#660000"  # MuSigma Burgundy/Red
MUSIGMA_SECONDARY = "#FF6A00"  # MuSigma Orange
MUSIGMA_LIGHT = "#FFF5F5"
MUSIGMA_DARK = "#4D0000"
MUSIGMA_CARD_BG = "#F8F9FA"
MUSIGMA_TABLE_HEADER = "#660000"
MUSIGMA_TABLE_ROW = "#FFF5F5"

# Apply custom CSS for MuSigma branding with animations
st.markdown(f"""
<style>
    /* Main background */
    .stApp {{
        background-color: #FFFFFF;
    }}
    
    /* Main header with fade-in animation */
    .main-header {{
        background-color: {MUSIGMA_PRIMARY};
        padding: 1.5rem 2rem;
        margin: -2rem -2rem 2rem -2rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        min-height: 80px;
        animation: fadeIn 1s ease-in;
    }}
    
    .brand-title {{
        color: white;
        font-size: 2.2rem;
        font-weight: bold;
        margin: 0;
        font-family: 'Arial', sans-serif;
    }}
    
    .brand-subtitle {{
        color: {MUSIGMA_LIGHT};
        font-size: 1rem;
        margin: 0;
        font-family: 'Arial', sans-serif;
    }}
    
    .logo-space {{
        width: 120px;
        height: 50px;
        background: rgba(255,255,255,0.1);
        border: 2px dashed rgba(255,255,255,0.3);
        border-radius: 5px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 0.8rem;
    }}
    
    /* Sidebar styling */
    .css-1d391kg {{
        background-color: {MUSIGMA_PRIMARY};
    }}
    
    /* Navigation items */
    .st-bb {{
        background-color: transparent;
    }}
    
    .st-bc {{
        background-color: {MUSIGMA_DARK};
        border-radius: 5px;
        margin: 2px 0;
    }}
    
    /* Radio button labels */
    .st-bh, .st-bi {{
        color: white;
        font-weight: 500;
    }}
    
    .st-eb {{
        background-color: {MUSIGMA_SECONDARY};
    }}
    
    /* Headers */
    h1, h2, h3 {{
        color: {MUSIGMA_PRIMARY};
        font-family: 'Arial', sans-serif;
    }}
    
    /* Consistent KPI Cards styling - FIXED EQUAL SIZE */
    .kpi-card {{
        background-color: {MUSIGMA_CARD_BG};
        padding: 1.5rem 1rem !important;
        border-radius: 10px;
        border-left: 4px solid {MUSIGMA_PRIMARY};
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        height: 140px !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
        align-items: center !important;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        margin-bottom: 1rem;
    }}
    
    .kpi-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.15);
    }}
    
    .kpi-label {{
        font-size: 0.9rem !important;
        font-weight: 600 !important;
        color: {MUSIGMA_PRIMARY} !important;
        text-align: center !important;
        margin-bottom: 0.5rem !important;
    }}
    
    .kpi-value {{
        font-size: 1.8rem !important;
        font-weight: bold !important;
        color: {MUSIGMA_DARK} !important;
        text-align: center !important;
        margin: 0 !important;
    }}
    
    .kpi-delta {{
        font-size: 0.9rem !important;
        text-align: center !important;
        margin-top: 0.5rem !important;
    }}
    
    /* Override Streamlit metric styling */
    [data-testid="stMetric"] {{
        background-color: transparent !important;
        padding: 0 !important;
        border: none !important;
        box-shadow: none !important;
        min-height: auto !important;
    }}
    
    [data-testid="stMetricValue"] {{
        font-size: 1.8rem !important;
    }}
    
    [data-testid="stMetricLabel"] {{
        font-size: 0.9rem !important;
    }}
    
    /* Dropdown styling - White text on MuSigma color */
    .stSelectbox > div > div {{
        background-color: {MUSIGMA_PRIMARY} !important;
        color: white !important;
        border: 2px solid {MUSIGMA_PRIMARY} !important;
        border-radius: 5px !important;
    }}
    
    .stSelectbox > div > div > div {{
        color: white !important;
    }}
    
    /* Dropdown options */
    .st-bx {{
        background-color: white !important;
        border: 2px solid {MUSIGMA_PRIMARY} !important;
    }}
    
    .st-bx > div {{
        color: {MUSIGMA_PRIMARY} !important;
    }}
    
    .st-bx > div:hover {{
        background-color: {MUSIGMA_LIGHT} !important;
    }}
    
    /* Table styling */
    .musigma-table {{
        background-color: white;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        animation: slideInUp 0.5s ease-out;
    }}
    
    .musigma-table thead th {{
        background-color: {MUSIGMA_TABLE_HEADER} !important;
        color: white !important;
        font-weight: bold !important;
        text-align: center !important;
        padding: 12px 8px !important;
    }}
    
    .musigma-table tbody tr:nth-child(even) {{
        background-color: {MUSIGMA_TABLE_ROW} !important;
    }}
    
    .musigma-table tbody tr:nth-child(odd) {{
        background-color: white !important;
    }}
    
    .musigma-table tbody tr:hover {{
        background-color: #FFE6E6 !important;
        transition: background-color 0.3s ease;
    }}
    
    .musigma-table tbody td {{
        text-align: center !important;
        padding: 10px 8px !important;
        border: none !important;
    }}
    
    /* Progress bar styling */
    .stProgress > div > div > div {{
        background-color: {MUSIGMA_SECONDARY};
    }}
    
    /* Animations */
    @keyframes fadeIn {{
        from {{ opacity: 0; }}
        to {{ opacity: 1; }}
    }}
    
    @keyframes slideInUp {{
        from {{ 
            opacity: 0;
            transform: translateY(20px);
        }}
        to {{ 
            opacity: 1;
            transform: translateY(0);
        }}
    }}
    
    @keyframes pulse {{
        0% {{ transform: scale(1); }}
        50% {{ transform: scale(1.05); }}
        100% {{ transform: scale(1); }}
    }}
    
    .pulse-animation {{
        animation: pulse 2s infinite;
    }}
    
    /* Chart container animations */
    .chart-container {{
        animation: fadeIn 0.8s ease-in;
    }}
    
    /* Explanation text alignment */
    .explanation-text {{
        text-align: justify;
        margin-top: 10px;
        padding: 10px;
        background-color: {MUSIGMA_LIGHT};
        border-radius: 5px;
        border-left: 3px solid {MUSIGMA_PRIMARY};
    }}
    
    /* Equal column height for KPI containers */
    .kpi-column {{
        display: flex;
        flex-direction: column;
        height: 100%;
    }}
</style>
""", unsafe_allow_html=True)

# ---------- Configuration & Data Path ----------

#  IMPORTANT: Replace with your actual path if needed
ABSOLUTE_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "attendance.xlsx") 

DATA_FILE = ABSOLUTE_DATA_PATH
if not os.path.exists(ABSOLUTE_DATA_PATH):
    CURRENT_DIR_PATH = os.path.join(os.getcwd(), "attendance.xlsx")
    if os.path.exists(CURRENT_DIR_PATH):
        DATA_FILE = CURRENT_DIR_PATH
    else:
        DATA_FILE = ABSOLUTE_DATA_PATH 

N_CLUSTERS = 4 
RANDOM_STATE = 42 

# Column constants
COL_FAKEID = "Fake ID"
COL_ACCOUNT = "Account code" 
COL_DESIGNATION = "Designation"
COL_RECRUITMENT = "Recruitment Type"
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

# --- Utility functions ---

@st.cache_data
def parse_time_or_duration(x):
    if pd.isna(x):
        return np.nan
    try:
        td = pd.to_timedelta(x)
        return td.total_seconds() / 3600.0
    except Exception:
        pass
    try:
        t = pd.to_datetime(x).time()
        return t.hour + t.minute / 60.0 + t.second / 3600.0
    except Exception:
        return np.nan

def safe_mean(series):
    return float(series.dropna().mean()) if len(series.dropna())>0 else np.nan

@st.cache_data
def load_and_prep(path):
    try:
        df = pd.read_excel(path)
    except FileNotFoundError:
        st.error(f"Error: Data file not found at **{path}**. Please ensure the file exists at this exact location.")
        return pd.DataFrame()
    
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    
    column_map = {}
    defined_cols = [COL_ACCOUNT, COL_DESIGNATION, COL_RECRUITMENT, COL_UNBILLED, COL_FAKEID, COL_IN, COL_OUT, 
                    COL_OFFICE, COL_BAY, COL_BREAK, COL_CAFE, COL_OOO, COL_HALF, COL_FULL, 
                    COL_ONLINE, COL_EXEMPT, COL_UNALLOC]
                    
    for defined_col in defined_cols:
        matching_col = next((c for c in df.columns if c.lower() == defined_col.lower()), None)
        if matching_col:
            column_map[defined_col] = matching_col
            
    df.rename(columns={v: k for k, v in column_map.items() if k != v}, inplace=True)
    
    if COL_ACCOUNT not in df.columns or COL_FAKEID not in df.columns:
        st.error(f"FATAL ERROR: Could not find '{COL_ACCOUNT}' or '{COL_FAKEID}' in the data. Please check your Excel column names.")
        return pd.DataFrame()
        
    for col in [COL_IN, COL_OUT]:
        if col in df.columns:
            df[col + "_hr"] = df[col].apply(parse_time_or_duration)

    for col in [COL_OFFICE, COL_BAY, COL_BREAK, COL_CAFE, COL_OOO]:
        if col in df.columns:
            df[col + "_hrs_float"] = df[col].apply(parse_time_or_duration)

    for col in [COL_HALF, COL_FULL, COL_ONLINE]:
        if col in df.columns:
            df[col + "_num"] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    
    df["unallocated_flag"] = df.get(COL_UNALLOC, pd.Series()).astype(str).str.lower().isin(["yes", "true", "1"]).astype(int)
    df["exempt_flag"] = df.get(COL_EXEMPT, pd.Series()).astype(str).str.lower().isin(["yes", "true", "1"]).astype(int)

    df['Billing Status'] = df.get(COL_UNBILLED, pd.Series()).astype(str).str.lower().str.contains("unbilled", na=False).map({True: 'Unbilled', False: 'Billed'})
    df['Leaves'] = df.get(COL_HALF + "_num", 0) + df.get(COL_FULL + "_num", 0)
    
    df["break_bay_ratio"] = df.apply(lambda r: (r.get("Avg. Break hrs_hrs_float", np.nan) / max(0.001, r.get("Avg. Bay hrs_hrs_float", np.nan))) if not pd.isna(r.get("Avg. Break hrs_hrs_float", np.nan)) else np.nan, axis=1)
    df["ooo_bay_ratio"] = df.apply(lambda r: (r.get("Avg. OOO hrs_hrs_float", np.nan) / max(0.001, r.get("Avg. Bay hrs_hrs_float", np.nan))) if not pd.isna(r.get("Avg. OOO hrs_hrs_float", np.nan)) else np.nan, axis=1)
    
    numeric_cols = [c for c in df.columns if c.endswith("_hrs_float") or c.endswith("_hr") or c.endswith("_num") or c in ["unallocated_flag", "exempt_flag", "break_bay_ratio", "ooo_bay_ratio"]] 
    for c in numeric_cols: 
        if c in df.columns: 
            median = df[c].median(skipna=True) 
            df[c] = df[c].fillna(median)

    return df

@st.cache_data(show_spinner="Running Anomaly Detection...")
def run_models(df_data):
    if df_data.empty or len(df_data) < 2:
        df_data["anomaly_flag"] = 0
        df_data["cluster"] = np.nan
        return df_data
        
    FEATURES = [] 
    for c in ["Avg. Bay hrs_hrs_float", "Avg. Break hrs_hrs_float", "Avg. OOO hrs_hrs_float", "Avg. Cafeteria hrs_hrs_float", 
                "Avg. Office hrs_hrs_float", "break_bay_ratio", "ooo_bay_ratio", "Half-Day leave_num", "Full-Day leave_num"]: 
        if c in df_data.columns: 
            FEATURES.append(c) 

    if not FEATURES: 
        df_data["anomaly_flag"] = 0
        df_data["cluster"] = np.nan
        return df_data

    scaler = StandardScaler()
    X = scaler.fit_transform(df_data[FEATURES].values)
    
    # K-Means (Kept for model context, even if not displayed)
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init='auto') 
    kmeans.fit(X) 
    df_data["cluster"] = kmeans.predict(X) 
    
    # Isolation Forest for Anomaly Detection
    iso = IsolationForest(contamination=0.05, random_state=RANDOM_STATE) 
    iso.fit(X) 
    df_data["anomaly_flag"] = np.where(iso.predict(X) == -1, 1, 0)
    
    return df_data

def get_employee_row(emp_id, df_data): 
    if df_data.empty: 
        return None 
    try: 
        emp_id_int = int(emp_id) 
    except ValueError: 
        return None 

    df_ids = pd.to_numeric(df_data[COL_FAKEID], errors="coerce") 

    subset = df_data[df_ids == emp_id_int] 
    if subset.empty: 
        return None 
    return subset.iloc[0] 

# --- Custom KPI Card Function ---
def create_kpi_card(label, value, delta=None, delta_color="normal"):
    """Create a custom KPI card with consistent sizing"""
    delta_html = ""
    if delta:
        delta_color_class = "stMetricDelta" 
        if delta_color == "inverse":
            delta_color_class = "stMetricDelta stMetricDelta--inverse"
        delta_html = f'<div class="kpi-delta {delta_color_class}">{delta}</div>'
    
    card_html = f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>
    """
    return card_html

# --- Plotly Charting Functions with MuSigma Colors ---

def create_bar_chart(title, labels, values):
    df_chart = pd.DataFrame({
        'Metric': labels,
        'Hours': values
    })
    fig = px.bar(df_chart, x='Metric', y='Hours', title=title, template="plotly_white", 
                 color='Hours', color_continuous_scale=[MUSIGMA_LIGHT, MUSIGMA_PRIMARY])
    fig.update_layout(xaxis={'categoryorder':'array', 'categoryarray':labels}, yaxis_title="Average Hours")
    return fig

def create_histogram_comparison(df_full, emp_row, metric_col, title, x_label):
    fig = px.histogram(df_full, x=metric_col, title=title, opacity=0.8, nbins=20, histnorm='percent',
                      color_discrete_sequence=[MUSIGMA_PRIMARY])
    
    if emp_row is not None and metric_col in emp_row:
        emp_value = emp_row[metric_col]
        if pd.notna(emp_value):
            fig.add_vline(x=emp_value, line_dash="dash", line_color=MUSIGMA_SECONDARY, line_width=2,
                          annotation_text=f"Employee: {emp_value:.2f}",
                          annotation_position="top right",
                          annotation_font_color=MUSIGMA_SECONDARY)
    
    fig.update_layout(showlegend=False, template="plotly_white", 
                      xaxis_title=x_label, yaxis_title="Percentage of Employees")
    return fig

def create_stacked_bar_distribution(data, title):
    dist_df = data.groupby([COL_ACCOUNT, COL_DESIGNATION]).size().reset_index(name='Employee Count')
    fig = px.bar(dist_df, x=COL_ACCOUNT, y='Employee Count', color=COL_DESIGNATION, title=title, 
                 template="plotly_white", height=400,
                 color_discrete_sequence=[MUSIGMA_PRIMARY, MUSIGMA_SECONDARY, MUSIGMA_DARK, '#CC0000', '#990000'])
    return fig

def create_pie_unbilled_global(data, title):
    billing_counts = data['Billing Status'].value_counts().reset_index()
    billing_counts.columns = ['Billing Status', 'Count']
    fig = px.pie(billing_counts, names='Billing Status', values='Count', title=title, 
                 color_discrete_map={'Billed': MUSIGMA_PRIMARY, 'Unbilled': MUSIGMA_SECONDARY}, 
                 template="plotly_white", hole=.3, height=400)
    return fig

def create_stacked_bar_productivity(data, title):
    work_hrs_cols = ["Avg. Bay hrs_hrs_float", "Avg. Break hrs_hrs_float", "Avg. Cafeteria hrs_hrs_float", "Avg. OOO hrs_hrs_float"]
    melted_df = data[[COL_DESIGNATION] + work_hrs_cols].melt(id_vars=COL_DESIGNATION, var_name='Time Category', value_name='Hours')
    avg_df = melted_df.groupby([COL_DESIGNATION, 'Time Category'])['Hours'].mean().reset_index()
    fig = px.bar(avg_df, x=COL_DESIGNATION, y='Hours', color='Time Category', title=title, 
                 template="plotly_white", height=450,
                 color_discrete_map={
                     "Avg. Bay hrs_hrs_float": MUSIGMA_PRIMARY,
                     "Avg. Break hrs_hrs_float": MUSIGMA_SECONDARY,
                     "Avg. Cafeteria hrs_hrs_float": "#CC0000",
                     "Avg. OOO hrs_hrs_float": MUSIGMA_DARK
                 })
    return fig

def create_boxplot_shift_patterns(data, title):
    time_df = data[[COL_DESIGNATION, COL_IN + '_hr', COL_OUT + '_hr']].rename(columns={COL_IN + '_hr': 'Avg. In Time', COL_OUT + '_hr': 'Avg. Out Time'})
    melted_time_df = time_df.melt(id_vars=COL_DESIGNATION, var_name='Metric', value_name='Hour of Day')
    fig = px.box(melted_time_df, x=COL_DESIGNATION, y='Hour of Day', color='Metric', title=title, 
                 template="plotly_white", height=500,
                 color_discrete_sequence=[MUSIGMA_PRIMARY, MUSIGMA_SECONDARY])
    return fig

# --------------------------------------------------------------------------------
# ---------- MODULE FUNCTIONS ----------

def executive_overview(df_raw):
    # Custom header for this page
    st.markdown(f"""
    <div class="main-header">
        <div>
            <div class="brand-title">MuTaskSpark</div>
            <div class="brand-subtitle">Executive Overview </div>
        </div>
        <div class="logo-space">
            Mu Sigma
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.header("Select Filters")
    
    col_f1, col_f2 = st.columns(2)
    
    accounts = ['All'] + sorted(df_raw[COL_ACCOUNT].unique().tolist())
    designations = ['All'] + sorted(df_raw[COL_DESIGNATION].unique().tolist())
    
    with col_f1:
        selected_account = st.selectbox("Filter by Account Code:", options=accounts, key='eo_account_filter')

    with col_f2:
        selected_designation = st.selectbox("Filter by Designation:", options=designations, key='eo_designation_filter')

    
    # --- Filter Logic ---
    filtered_df = df_raw.copy()
    
    is_filtered = (selected_account != 'All') or (selected_designation != 'All')
    
    if selected_account != 'All':
        filtered_df = filtered_df[filtered_df[COL_ACCOUNT] == selected_account]
    if selected_designation != 'All':
        filtered_df = filtered_df[filtered_df[COL_DESIGNATION] == selected_designation]
        
    
    if not is_filtered:
        st.info("Please select a specific Account Code or Designation filter to view the Executive Metrics and Charts.")
        return
        
    if filtered_df.empty:
        st.warning("No data found for the selected filters.")
        return

    
    # --- KPI CALCULATIONS ---
    total_employees = len(filtered_df)
    avg_bay = filtered_df.get(COL_BAY + "_hrs_float", pd.Series(0.0)).mean(skipna=True)
    total_unbilled = filtered_df['Billing Status'].value_counts().get('Unbilled', 0)
    billed_count = filtered_df['Billing Status'].value_counts().get('Billed', 0)
    billed_pct = (billed_count / max(1, total_employees)) * 100
    total_full = filtered_df.get(COL_FULL + "_num", pd.Series(0.0)).sum(skipna=True)
    total_half = filtered_df.get(COL_HALF + "_num", pd.Series(0.0)).sum(skipna=True)
    total_leave_days = total_full + (total_half / 2)
    
    
    # --- KPI CARDS - EQUAL SIZE ---
    st.divider()
    st.subheader("Key Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(create_kpi_card("👥 Total Employees", f"{total_employees:,}"), unsafe_allow_html=True)
    with col2:
        st.markdown(create_kpi_card("🎯 Avg. Focused (Bay) Hrs", f"{avg_bay:.2f} hrs"), unsafe_allow_html=True)
    with col3:
        st.markdown(create_kpi_card("💰 Billed Percentage", f"{billed_pct:.1f}%", 
                                   delta=f"Unbilled: {(100 - billed_pct):.1f}%", delta_color="inverse"), unsafe_allow_html=True)
    with col4:
        st.markdown(create_kpi_card("📅 Total Leave Days", f"{total_leave_days:,.1f}"), unsafe_allow_html=True)
          
    st.divider()
    
    # Charts Section 1
    st.subheader("Workforce and Billing Analysis")
    col_dist, col_billing = st.columns([2, 1])
    with col_dist:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.plotly_chart(create_stacked_bar_distribution(filtered_df, "Workforce Distribution: Account vs Designation"), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col_billing:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.plotly_chart(create_pie_unbilled_global(filtered_df, "Billing Status Distribution"), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Properly aligned explanation
    st.markdown("""
    <div class="explanation-text">
    <strong>Explanation: Workforce Distribution & Billing Status</strong> 🏢<br>
    - <strong>Left Chart (Bar):</strong> Shows the <strong>headcount distribution</strong> across various <strong>Account Codes</strong> (X-axis), segmented by <strong>Designation</strong> (color). This helps visualize the staffing mix and capacity distribution.
    - <strong>Right Chart (Pie):</strong> Illustrates the overall <strong>financial risk</strong> by showing the percentage breakdown of employees who are <strong>Billed</strong> versus <strong>Unbilled</strong> within the current filter selection.
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Charts Section 2
    st.subheader("Productivity and Shift Patterns")
    col_prod_bar, col_shift = st.columns(2)
    with col_prod_bar:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.plotly_chart(create_stacked_bar_productivity(filtered_df, "Avg. Time Allocation by Designation"), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col_shift:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.plotly_chart(create_boxplot_shift_patterns(filtered_df, "Shift Patterns: Avg. In/Out Time by Designation"), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Properly aligned explanation for productivity and shift patterns
    st.markdown("""
    <div class="explanation-text">
    <strong>Explanation: Productivity & Shift Patterns</strong> ⏱️<br>
    - <strong>Left Chart (Bar):</strong> Compares the <strong>average time allocation</strong> across different <strong>Designations</strong> (X-axis). It breaks down time into <strong>Focused (Bay), Break, Cafeteria, and OOO (Out of Office)</strong> hours to assess where effort is concentrated.
    - <strong>Right Chart (Box Plot):</strong> Displays the <strong>distribution (range and median)</strong> of employee <strong>Avg. In Time</strong> and <strong>Avg. Out Time</strong> by Designation. This highlights consistency and potential issues with early starts or late finishes.
    </div>
    """, unsafe_allow_html=True)

    if total_unbilled > 0:
        st.warning(f"⚠️ **Action Required:** There are **{total_unbilled}** employees currently categorized as **Unbilled** in this filtered group.")

# ---------------------- Employee Analysis ---------------------
def time_to_hours(t):
    if pd.isna(t):
        return 0.0
    if isinstance(t, str):
        try:
            h, m = map(int, t.split(":"))
            return h + m / 60
        except:
            return 0.0
    if isinstance(t, pd.Timestamp):
        return t.hour + t.minute / 60
    return float(t)

def calculate_z_scores(emp_row, df):
    z_scores = {}
    metrics = {
        "Bay": COL_BAY + "_hrs_float",
        "Break": COL_BREAK + "_hrs_float",
        "OOO": COL_OOO + "_hrs_float",
        "Office": COL_OFFICE + "_hrs_float"
    }
    for name, col in metrics.items():
        if col in df.columns:
            mu = df[col].mean()
            sigma = df[col].std(ddof=0)
            if sigma == 0: sigma = 1.0 
            z_scores[name] = (emp_row.get(col, mu) - mu) / sigma
    return z_scores

def donut_chart(emp_row):
    bay = emp_row.get(COL_BAY+"_hrs_float",0)
    brk = emp_row.get(COL_BREAK+"_hrs_float",0)
    cafe = emp_row.get(COL_CAFE+"_hrs_float",0)
    ooo = emp_row.get(COL_OOO+"_hrs_float",0)
    office = emp_row.get(COL_OFFICE+"_hrs_float",0)
    other = max(0, office - (bay+brk+cafe+ooo))

    donut_df = pd.DataFrame({
        "Activity": ["Focused (Bay)","Break","Cafeteria","OOO","Other"],
        "Hours": [bay, brk, cafe, ooo, other]
    })
    donut_df = donut_df[donut_df['Hours'] > 0.01]
    
    fig_donut = px.pie(donut_df, names="Activity", values="Hours", hole=0.45,
                        title=f"Time Allocation Distribution (Total: {office:.1f} hrs)",
                        color="Activity",
                        color_discrete_map={
                            "Focused (Bay)": MUSIGMA_PRIMARY, 
                            "Break": MUSIGMA_SECONDARY, 
                            "Cafeteria": "#CC0000", 
                            "OOO": MUSIGMA_DARK,
                            "Other":"#A5A58D"
                        })
    fig_donut.update_traces(textinfo="percent+label")
    return fig_donut

def comparison_chart(df, emp_row):
    metrics = [COL_BAY + "_hrs_float", COL_BREAK + "_hrs_float", COL_CAFE + "_hrs_float", COL_OOO + "_hrs_float"]
    emp_vals = [emp_row.get(m, 0) for m in metrics]
    comp_vals = [df[m].mean() for m in metrics]
    df_compare = pd.DataFrame({
        "Metric":["Bay","Break","Cafeteria","OOO"],
        "Employee":emp_vals,
        "Company Avg":comp_vals
    })
    fig_compare = go.Figure()
    fig_compare.add_trace(go.Bar(x=df_compare["Metric"], y=df_compare["Employee"], name="Employee", marker_color=MUSIGMA_PRIMARY))
    fig_compare.add_trace(go.Bar(x=df_compare["Metric"], y=df_compare["Company Avg"], name="Company Avg", marker_color=MUSIGMA_SECONDARY))
    fig_compare.update_layout(title="Time Allocation Comparison to Company Average (Hours)", barmode="group", yaxis_title="Hours", template="plotly_white")
    return fig_compare

def hours_to_hhmm(hours):
    if pd.isna(hours):
        return "N/A"
    h = int(hours)
    m = int(round((hours - h) * 60))
    return f"{h:02d}:{m:02d} hrs"

def employee_level_analysis(df):
    # Custom header for this page
    st.markdown(f"""
    <div class="main-header">
        <div>
            <div class="brand-title">MuTaskSpark</div>
            <div class="brand-subtitle">Employee Performance Analytics</div>
        </div>
        <div class="logo-space">
            MuSigma Logo
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.header("Select Employee Id👤")

    employee_ids = sorted(df[COL_FAKEID].astype(str).unique())
    select_options = ["--- Select an Employee ID ---"] + employee_ids
    
    emp_id = st.selectbox(
        "Employee ID", 
        select_options, 
        index=0, 
        key='employee_select'
    )
    
    if emp_id == "--- Select an Employee ID ---":
        st.info("Please select an Employee ID from the dropdown to view the detailed analysis.")
        return 
    
    emp_row = get_employee_row(emp_id, df)
    if emp_row is None:
        st.error(f"Employee ID {emp_id} not found in the dataset.")
        return

    # Extract data
    in_time_hr = emp_row.get(COL_IN + "_hr", np.nan)
    out_time_hr = emp_row.get(COL_OUT + "_hr", np.nan)
    bay = emp_row.get(COL_BAY + "_hrs_float", 0)
    brk = emp_row.get(COL_BREAK + "_hrs_float", 0)
    cafe = emp_row.get(COL_CAFE + "_hrs_float", 0)
    ooo = emp_row.get(COL_OOO + "_hrs_float", 0)
    office = emp_row.get(COL_OFFICE + "_hrs_float", 0)
    half = emp_row.get(COL_HALF + "_num", 0)
    full = emp_row.get(COL_FULL + "_num", 0)
    online_checkins = emp_row.get(COL_ONLINE + "_num", 0) 
    designation = emp_row.get(COL_DESIGNATION, "N/A")
    account = emp_row.get(COL_ACCOUNT, "N/A")
    billing_status = emp_row.get("Billing Status", "N/A")

    # KPI Cards - EQUAL SIZE
    st.subheader("Employee Metrics")

    col_r1_1, col_r1_2, col_r1_3, col_r1_4 = st.columns(4)
    with col_r1_1:
        st.markdown(create_kpi_card("🧑‍💼 Designation", designation), unsafe_allow_html=True)
    with col_r1_2:
        st.markdown(create_kpi_card("🏢 Account Code", account), unsafe_allow_html=True)
    with col_r1_3:
        st.markdown(create_kpi_card("💵 Billing Status", billing_status), unsafe_allow_html=True)
    with col_r1_4:
        st.markdown(create_kpi_card("💻 Online Check-ins", f"{online_checkins:.0f}"), unsafe_allow_html=True)

    col_r2_1, col_r2_2, col_r2_3, col_r2_4 = st.columns(4)
    with col_r2_1:
        st.markdown(create_kpi_card("⏰ Avg In Time", hours_to_hhmm(in_time_hr)), unsafe_allow_html=True)
    with col_r2_2:
        st.markdown(create_kpi_card("🕔 Avg Out Time", hours_to_hhmm(out_time_hr)), unsafe_allow_html=True)
    with col_r2_3:
        st.markdown(create_kpi_card("📅 Full-Day Leaves", f"{full:.1f}"), unsafe_allow_html=True)
    with col_r2_4:
        st.markdown(create_kpi_card("🩳 Half-Day Leaves", f"{half:.1f}"), unsafe_allow_html=True)

    st.divider()

    # Charts
    st.subheader("Behavioral Analysis and Distribution 📊")
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.plotly_chart(donut_chart(emp_row), use_container_width=True, key=f'donut_{emp_id}') 
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.plotly_chart(comparison_chart(df, emp_row), use_container_width=True, key=f'compare_{emp_id}')
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.divider()

    # Engagement & Findings
    z_scores = calculate_z_scores(emp_row, df)
    engagement_score = ((bay - (brk + cafe + ooo)) / office * 100) if office > 0 else 0
    engagement_score = max(0, engagement_score)

    st.subheader("Key Findings 🔍")
    
    if engagement_score < 60:
        st.warning(f"🔴 **Engagement Score:** **{engagement_score:.1f}%** - **Low Productivity Ratio.**")
    elif engagement_score < 80:
        st.info(f"🟠 **Engagement Score:** **{engagement_score:.1f}%** - **Moderate Productivity Ratio.**")
    else:
        st.success(f"🟢 **Engagement Score:** **{engagement_score:.1f}%** - **High Productivity Ratio.**")
    
    st.markdown("**Engagement score:** = (Avg Bay Hrs − (Avg Break Hrs + Avg Cafeteria Hrs + Avg OOO Hrs)) / Avg Office Hrs × 100")
    
    st.markdown("#### Performance Analysis:")
    
    bay_z = z_scores.get('Bay', 0)
    if bay_z < -1.5:
        st.markdown(f"- **Focused Hours (Bay):** Critically low ({bay:.1f} hrs, >1.5 SD below avg). **Requires immediate attention.**")
    elif bay_z > 1.5:
        st.markdown(f"- **Focused Hours (Bay):** Exceptionally high ({bay:.1f} hrs, >1.5 SD above avg).")
    else:
        st.markdown(f"- **Focused Hours (Bay):** Normal relative to the peer group.")

    brk_z = z_scores.get('Break', 0)
    ooo_z = z_scores.get('OOO', 0)
    if brk_z > 1.5:
        st.markdown(f"- **Break Time:** Significantly high ({brk:.1f} hrs, >1.5 SD above avg). **Strongly suggests time management coaching.**")
    elif ooo_z > 1.5:
        st.markdown(f"- **OOO Time:** Significantly high ({ooo:.1f} hrs). **Potential overload of meetings or external commitments.**")
    else:
        st.markdown(f"- **Break/OOO Time:** Generally within the expected range.")

    if office > 10.5 and bay < 7 and engagement_score < 70:
        st.markdown("- **Low Efficiency:** Long working days ({office:.1f} hrs) are not translating to focused output. **High risk of burnout and low ROI.**")
    if billing_status == "Unbilled":
         st.markdown("⚠️ **Billing Alert:** Employee is currently categorized as **Unbilled**. This requires an immediate business/allocation review.")
        
    if full > 5 or half > 15:
        st.markdown(f"- **Leave Frequency:** **{full:.0f}** full days and **{half:.0f}** half days are unusually high. **Check for stress or consistent availability issues.**")
        
    st.divider()

    # Recommendations
    st.subheader("Recommended Action Plan 💡")
    recommendations = []

    if engagement_score < 60:
        recommendations.append("**A1. Strategic Intervention:** Schedule an **immediate, non-punitive** 1-on-1 to identify workflow friction and environmental distractions. **Goal:** Establish a baseline focus routine.")
    elif office > 10.5 and bay < 7:
        recommendations.append("**A2. Efficiency Audit:** Conduct a deep dive into the daily task list to minimize context-switching and ensure alignment between long hours and focused work delivery.")
    elif engagement_score > 85:
        recommendations.append("**A3. Positive Reinforcement:** Officially recognize this employee's discipline. Leverage their work pattern as a **Success Benchmark** for the team.")

    if brk_z > 1.5:
        recommendations.append("⚠️ **B1. Time Management Coaching:** Implement a structured 1-week goal to reduce average break time by 10-15 minutes, focusing on scheduled recovery breaks.")
    if ooo_z > 1.5:
        recommendations.append("🗓️ **B2. Meeting Review:** Review the employee's calendar for the last month. Challenge the necessity of non-essential meetings and establish 'No Meeting Focus Blocks'.")

    if not pd.isna(in_time_hr) and in_time_hr > 10.5: 
        recommendations.append("⏰ **C1. Attendance Consistency:** Address the trend of late check-ins to ensure seamless morning collaboration, especially if core hours are mandated.")
    if billing_status == "Unbilled":
        recommendations.append("💰 **C2. Allocation Resolution:** **URGENT** - Hand off to Project Management/Finance to resolve the unbilled status within 48 hours.")

    if full > 5 or half > 15:
        recommendations.append("🩺 **D1. HR/Wellness Flag:** Flag the high volume of leaves for HR review to check for potential stress, health issues, or dependency patterns.")

    if not recommendations:
        st.info("No critical actions required. Maintain current monitoring levels.")
    else:
        for rec in recommendations:
            st.markdown(f"- {rec}")

    st.divider()

def account_level_analysis(df_full):
    # Custom header for this page
    st.markdown(f"""
    <div class="main-header">
        <div>
            <div class="brand-title">MuTaskSpark</div>
            <div class="brand-subtitle">Account Performance Analytics</div>
        </div>
        <div class="logo-space">
            MuSigma Logo
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.header("Select Account 🏢")

    # Get unique account codes
    account_codes = sorted(df_full[COL_ACCOUNT].dropna().unique().tolist())
    
    if not account_codes:
        st.error("No account codes found in the dataset.")
        return
        
    selected_account = st.selectbox(
        "Select Account Code:", 
        options=["--- Select Account ---"] + account_codes,
        key='account_select'
    )
    
    if selected_account == "--- Select Account ---":
        st.info("Please select an Account Code to view detailed analysis.")
        return

    # Filter data for selected account
    account_df = df_full[df_full[COL_ACCOUNT] == selected_account]
    
    if account_df.empty:
        st.warning(f"No data found for account: {selected_account}")
        return

    # --- KPI Calculations ---
    total_employees = len(account_df)
    avg_office_hrs = account_df.get(COL_OFFICE + "_hrs_float", pd.Series(0.0)).mean(skipna=True)
    avg_bay_hrs = account_df.get(COL_BAY + "_hrs_float", pd.Series(0.0)).mean(skipna=True)
    avg_break_hrs = account_df.get(COL_BREAK + "_hrs_float", pd.Series(0.0)).mean(skipna=True)
    avg_cafe_hrs = account_df.get(COL_CAFE + "_hrs_float", pd.Series(0.0)).mean(skipna=True)
    avg_ooo_hrs = account_df.get(COL_OOO + "_hrs_float", pd.Series(0.0)).mean(skipna=True)
    
    # Billing analysis
    unbilled_count = account_df['Billing Status'].value_counts().get('Unbilled', 0)
    billed_count = account_df['Billing Status'].value_counts().get('Billed', 0)
    billed_pct = (billed_count / max(1, total_employees)) * 100
    
    # Leave analysis
    total_full_leaves = account_df.get(COL_FULL + "_num", pd.Series(0.0)).sum(skipna=True)
    total_half_leaves = account_df.get(COL_HALF + "_num", pd.Series(0.0)).sum(skipna=True)
    avg_leaves_per_emp = (total_full_leaves + total_half_leaves/2) / max(1, total_employees)
    
    # Anomaly analysis
    anomaly_count = account_df["anomaly_flag"].sum()
    anomaly_rate = (anomaly_count / max(1, total_employees)) * 100

    # --- KPI Cards - EQUAL SIZE ---
    st.subheader("Account Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(create_kpi_card("👥 Total Employees", f"{total_employees:,}"), unsafe_allow_html=True)
    with col2:
        st.markdown(create_kpi_card("💰 Billed Rate", f"{billed_pct:.1f}%", 
                                   delta=f"Unbilled: {unbilled_count}", delta_color="inverse"), unsafe_allow_html=True)
    with col3:
        st.markdown(create_kpi_card("⚠️ Anomaly Rate", f"{anomaly_rate:.1f}%", 
                                   delta=f"Flags: {anomaly_count}"), unsafe_allow_html=True)
    with col4:
        st.markdown(create_kpi_card("📅 Avg Leaves/Emp", f"{avg_leaves_per_emp:.1f}"), unsafe_allow_html=True)

    st.divider()

    # --- Productivity Metrics - EQUAL SIZE ---
    st.subheader("Productivity Analysis")
    
    col_prod1, col_prod2, col_prod3, col_prod4 = st.columns(4)
    
    with col_prod1:
        st.markdown(create_kpi_card("🏢 Avg Office Hrs", f"{avg_office_hrs:.1f} hrs"), unsafe_allow_html=True)
    with col_prod2:
        st.markdown(create_kpi_card("🎯 Avg Bay Hrs", f"{avg_bay_hrs:.1f} hrs"), unsafe_allow_html=True)
    with col_prod3:
        st.markdown(create_kpi_card("☕ Avg Break Hrs", f"{avg_break_hrs:.1f} hrs"), unsafe_allow_html=True)
    with col_prod4:
        st.markdown(create_kpi_card("🗣️ Avg OOO Hrs", f"{avg_ooo_hrs:.1f} hrs"), unsafe_allow_html=True)

    # Productivity Ratio
    productivity_ratio = (avg_bay_hrs / avg_office_hrs * 100) if avg_office_hrs > 0 else 0
    
    st.progress(min(productivity_ratio/100, 1.0), 
               text=f"Productivity Ratio: {productivity_ratio:.1f}% (Bay Hrs / Office Hrs)")

    st.divider()

    # --- Visualizations ---
    st.subheader("Visual Analytics")
    
    # Chart 1: Designation Distribution
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        if COL_DESIGNATION in account_df.columns:
            designation_counts = account_df[COL_DESIGNATION].value_counts()
            fig_designation = px.pie(
                values=designation_counts.values,
                names=designation_counts.index,
                title=f"Designation Distribution - {selected_account}",
                color_discrete_sequence=[MUSIGMA_PRIMARY, MUSIGMA_SECONDARY, MUSIGMA_DARK, '#CC0000']
            )
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.plotly_chart(fig_designation, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col_chart2:
        # Billing Status
        billing_data = account_df['Billing Status'].value_counts()
        fig_billing = px.bar(
            x=billing_data.index,
            y=billing_data.values,
            title="Billing Status Distribution",
            color=billing_data.index,
            color_discrete_map={'Billed': MUSIGMA_PRIMARY, 'Unbilled': MUSIGMA_SECONDARY}
        )
        fig_billing.update_layout(showlegend=False, xaxis_title="Billing Status", yaxis_title="Count")
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.plotly_chart(fig_billing, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Chart 2: Time Allocation by Designation
    if COL_DESIGNATION in account_df.columns:
        time_metrics = [COL_BAY + "_hrs_float", COL_BREAK + "_hrs_float", COL_OOO + "_hrs_float"]
        time_df = account_df[[COL_DESIGNATION] + time_metrics].groupby(COL_DESIGNATION).mean().reset_index()
        
        fig_time = px.bar(
            time_df, 
            x=COL_DESIGNATION,
            y=time_metrics,
            title="Average Time Allocation by Designation",
            barmode='group',
            color_discrete_sequence=[MUSIGMA_PRIMARY, MUSIGMA_SECONDARY, MUSIGMA_DARK]
        )
        fig_time.update_layout(xaxis_title="Designation", yaxis_title="Hours")
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.plotly_chart(fig_time, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    # --- Key Findings & Recommendations ---
    st.subheader("Key Findings")
    
    findings = []
    
    if billed_pct < 70:
        findings.append(f"🚨 **Critical Billing Issue:** Only {billed_pct:.1f}% employees are billed. High financial risk.")
    elif billed_pct < 85:
        findings.append(f"⚠️ **Billing Concern:** {billed_pct:.1f}% billing rate needs improvement.")
    else:
        findings.append(f"✅ **Healthy Billing:** {billed_pct:.1f}% billing rate is excellent.")
    
    if productivity_ratio < 50:
        findings.append(f"📉 **Low Productivity:** Only {productivity_ratio:.1f}% of office time is focused work.")
    elif productivity_ratio > 70:
        findings.append(f"🏆 **High Productivity:** {productivity_ratio:.1f}% focused work ratio is outstanding.")
    
    if avg_break_hrs > 1.5:
        findings.append(f"⏰ **High Break Time:** Average break duration of {avg_break_hrs:.1f} hrs may indicate inefficiency.")
    
    if anomaly_rate > 15:
        findings.append(f"🔴 **High Anomaly Rate:** {anomaly_rate:.1f}% of employees show unusual patterns.")
    elif anomaly_rate > 5:
        findings.append(f"🟡 **Moderate Anomaly Rate:** {anomaly_rate:.1f}% anomalies need monitoring.")
    
    if avg_leaves_per_emp > 8:
        findings.append(f"📅 **High Leave Rate:** {avg_leaves_per_emp:.1f} average leaves per employee.")

    for finding in findings:
        st.markdown(f"- {finding}")

    st.divider()

    # --- Recommendations ---
    st.subheader("Recommendations")
    
    recommendations = []
    
    if billed_pct < 70:
        recommendations.append("**Resource Reallocation:** Immediate review of unbilled resources and project assignments.")
    if productivity_ratio < 50:
        recommendations.append("**Productivity Workshop:** Implement focused work training and time management sessions.")
    if avg_break_hrs > 1.5:
        recommendations.append("**Break Management:** Establish structured break schedules and monitor adherence.")
    if anomaly_rate > 15:
        recommendations.append("**Deep Dive Analysis:** Investigate root causes of high anomaly rates with HR.")
    if avg_leaves_per_emp > 8:
        recommendations.append("**Employee Engagement:** Review workload and conduct wellness check-ins.")
    
    if not recommendations:
        recommendations.append("**Maintain Current Strategy:** Account performance is within acceptable parameters. Continue regular monitoring.")
    
    for i, rec in enumerate(recommendations, 1):
        st.markdown(f"{i}. {rec}")

    # --- Employee List ---
    st.divider()
    st.subheader("Account Employees")
    
    # Create a simplified employee list
    employee_cols = [COL_FAKEID, COL_DESIGNATION, 'Billing Status', COL_BAY + "_hrs_float", COL_OFFICE + "_hrs_float"]
    available_cols = [col for col in employee_cols if col in account_df.columns]
    
    if available_cols:
        display_df = account_df[available_cols].copy()
        display_df[COL_BAY + "_hrs_float"] = display_df[COL_BAY + "_hrs_float"].round(1)
        display_df[COL_OFFICE + "_hrs_float"] = display_df[COL_OFFICE + "_hrs_float"].round(1)
        
        # Apply MuSigma table styling
        st.markdown('<div class="musigma-table">', unsafe_allow_html=True)
        st.dataframe(
            display_df,
            use_container_width=True,
            height=300,
            hide_index=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------------------------------------
# ---------- MAIN APP EXECUTION ----------

try:
    df_raw = run_models(load_and_prep(DATA_FILE))
    if df_raw.empty:
        st.stop()
except Exception as e:
    st.error(f"Failed to load or process data: {e}")
    st.stop()

# --- SIDEBAR NAVIGATION ---
st.sidebar.markdown(f"""
<div style='background-color: {MUSIGMA_PRIMARY}; padding: 2rem 1rem; text-align: center; margin: -1rem -1rem 2rem -1rem;'>
    <h2 style='color: white; margin: 0; font-size: 1.5rem;'>MuTaskSpark</h2>
    <p style='color: {MUSIGMA_LIGHT}; margin: 0;'>Analytics Dashboard</p>
</div>
""", unsafe_allow_html=True)

PAGES = {
    "🏢 Executive Overview": executive_overview,
    "👤 Employee Analysis": employee_level_analysis,
    "📊 Account Analytics": account_level_analysis
}

st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to:", list(PAGES.keys()))

# Footer in sidebar
st.sidebar.markdown(f"""
<div style='margin-top: 3rem; padding: 1rem; background-color: {MUSIGMA_DARK}; border-radius: 0.5rem;'>
    <p style='color: white; text-align: center; margin: 0; font-size: 0.8rem;'>
        <b>MuSigma</b><br>
        Decision Sciences & Analytics
    </p>
</div>
""", unsafe_allow_html=True)

# RENDER SELECTED PAGE
page_func = PAGES[selection]
page_func(df_raw)