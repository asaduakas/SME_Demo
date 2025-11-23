import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Construction Fleet Parts Intelligence", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("synthetic_construction_parts_dataset.csv")

df = load_data()

st.sidebar.header("Filters")
machine_filter = st.sidebar.multiselect("Machine Type", df["machine_type"].unique())
category_filter = st.sidebar.multiselect("Part Category", df["part_category"].unique())
supplier_filter = st.sidebar.multiselect("Supplier", df["supplier"].unique())

filtered_df = df.copy()
if machine_filter:
    filtered_df = filtered_df[filtered_df["machine_type"].isin(machine_filter)]
if category_filter:
    filtered_df = filtered_df[filtered_df["part_category"].isin(category_filter)]
if supplier_filter:
    filtered_df = filtered_df[filtered_df["supplier"].isin(supplier_filter)]

# ---------- TAB LAYOUT ----------
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🤖 ML Predictions", "📦 Inventory Optimization", "💰 Cost Simulation"])

# =========================================
# TAB 1: OVERVIEW
# =========================================
with tab1:
    st.title("📊 AI-Powered Construction Parts Dashboard")
    st.write("A predictive maintenance & procurement intelligence demo for heavy construction machinery.")
    #KPIs
    total_parts = len(filtered_df)
    avg_failure = filtered_df["failure_rate_est"].mean()
    avg_lead_time = filtered_df["lead_time_days"].mean()
    avg_price = filtered_df["part_price_usd"].mean()

    st.markdown("---")

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    kpi1.metric("Total Parts", f"{total_parts:,}")
    kpi2.metric("Avg Failure Rate", f"{avg_failure:.2f}")
    kpi3.metric("Avg Lead Time (days)", f"{avg_lead_time:.1f}")
    kpi4.metric("Avg Part Price (USD)", f"${avg_price:.2f}")

    st.markdown("---")

    #Failure risk scoring
    st.subheader("📉 Parts Failure Probability Distribution")

    fig1 = px.histogram(filtered_df, 
                        x="failure_rate_est", 
                        nbins=20, 
                        title="Failure Rate Distribution", 
                        labels={"failure_rate_est":"Estimated Failure Probability"})
    st.plotly_chart(fig1, width="stretch")

    #Downtime impact x failure rate
    st.subheader("⚠️ High-Risk Parts (Failure × Downtime Impact)")
    filtered_df["risk_score"] = filtered_df["failure_rate_est"] * filtered_df["downtime_impact_hours"]

    fig2 = px.scatter(
        filtered_df,
        x = "failure_rate_est",
        y = "downtime_impact_hours",
        size = "risk_score",
        color = "part_category",
        hover_data = ["part_number", "supplier"],
        title = "Failure Probability vs Downtime Impact"
    )
    st.plotly_chart(fig2, width="stretch")

    #Supplier analysis
    st.subheader("📦 Supplier Lead Time Analysis")

    supplier_group = (
        filtered_df.groupby("supplier")
        .agg({"lead_time_days":"mean", "part_price_usd":"mean"})
        .reset_index()
    )

    fig3 = px.bar(
        supplier_group,
        x = "supplier",
        y = "lead_time_days",
        title = "Average Lead Time by Supplier (days)",
        labels = {"lead_time_days": "Lead Time (days)"}
    )

    st.plotly_chart(fig3, width='stretch')

    st.subheader("🔍 Top 10 Most Critical Parts (by Risk Score)")

    critical = filtered_df.sort_values("risk_score", ascending=False).head(10)
    st.dataframe(critical)


# =========================================
# TAB 2: ML PREDICTIONS (already finished in Step 1)
# =========================================
with tab2:
    #Target
    df["due_for_replacement"] = (df["operating_hours"] >= df["expected_replacement_interval_hours"]).astype(int)

    st.header("🤖 Predictive Failure Model: Due for Replacement")

    FEATURES = ["machine_type", "part_category", "supplier", "operating_hours", "part_price_usd", "failure_rate_est", "lead_time_days", "downtime_impact_hours"]

    X = df[FEATURES]
    y = df["due_for_replacement"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    categorical_features = ["machine_type", "part_category", "supplier"]
    numeric_features = [c for c in FEATURES if c not in categorical_features]

    #Preprocessing: one hot categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ],
        remainder="passthrough"
    )

    #Pipeline with RandomForest
    pipe = Pipeline([
        ("pre", preprocessor),
        ("clf", RandomForestClassifier(n_estimators=200, random_state=42))
    ])

    #Train
    with st.spinner("Training model..."):
        pipe.fit(X_train, y_train)

    #Predict
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:,1]

    acc = accuracy_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, y_proba)
    except:
        auc = None

    st.subheader("Model performance (on synthetic test set)")
    col_a, col_b = st.columns(2)
    col_a.metric("accuracy", f"{acc:.3f}")
    col_b.metric("ROC AUC", f"{auc:.3f}" if auc is not None else "N/A")

    #Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    st.text("Confusion matrix (rows:true, cols: predicted)")
    st.write(pd.DataFrame(cm, index=["True 0","True 1"], columns=["Pred 0","Pred 1"]))

    # Feature importance: compute approximate importances by transforming once
    # Extract feature names after one-hot encoding
    ohe = pipe.named_steps["pre"].named_transformers_["cat"]
    ohe_names = list(ohe.get_feature_names_out(categorical_features))
    feature_names = ohe_names + numeric_features
    importances = pipe.named_steps["clf"].feature_importances_

    fi_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    fi_df = fi_df.sort_values("importance", ascending=False).head(15)
    st.subheader("Top feature importances")
    st.bar_chart(fi_df.set_index("feature")["importance"])

    st.markdown("---")

    # --- Prediction UI ---
    st.subheader("🔍 Predict for a selected part")
    st.write("Pick a row from the filtered table to get model predictions, or enter custom values.")

    # Show filtered dataframe with selectbox for index
    st.write("Filtered dataset (first 200 rows)")
    st.dataframe(filtered_df.head(200))

    selected_idx = st.number_input("Enter index (0-based) from filtered table to predict (or -1 to skip)", min_value=-1, max_value=len(filtered_df)-1, value=-1, step=1)

    if selected_idx >= 0:
        # map filtered_df index to original df index
        chosen_row = filtered_df.reset_index().iloc[selected_idx]
        st.write("Selected row:")
        st.write(chosen_row[FEATURES])
        # prepare for prediction
        X_row = pd.DataFrame([chosen_row[FEATURES].to_dict()])
        prob = pipe.predict_proba(X_row)[:, 1][0]
        pred = pipe.predict(X_row)[0]
        st.metric("Predicted probability of being due", f"{prob:.2%}")
        st.write("Prediction (0 = not due, 1 = due):", int(pred))

    st.markdown("### Predict custom values")
    with st.form("predict_form"):
        mtype = st.selectbox("Machine Type", df["machine_type"].unique())
        pcat = st.selectbox("Part Category", df["part_category"].unique())
        supp = st.selectbox("Supplier", df["supplier"].unique())
        op_hours = st.number_input("Operating hours", min_value=0, value=1000)
        pprice = st.number_input("Part price (USD)", min_value=0.0, value=200.0)
        fail_rate = st.slider("Failure rate estimate (0-1)", 0.0, 1.0, float(df["failure_rate_est"].mean()))
        lead_days = st.number_input("Lead time (days)", min_value=0, value=14)
        down_hours = st.number_input("Downtime impact (hours)", min_value=0, value=40)
        submit = st.form_submit_button("Predict")

        if submit:
            X_new = pd.DataFrame([{
                "machine_type": mtype,
                "part_category": pcat,
                "supplier": supp,
                "operating_hours": op_hours,
                "part_price_usd": pprice,
                "failure_rate_est": fail_rate,
                "lead_time_days": lead_days,
                "downtime_impact_hours": down_hours
            }])
            prob = pipe.predict_proba(X_new)[:, 1][0]
            pred = int(pipe.predict(X_new)[0])
            st.metric("Predicted probability of being due", f"{prob:.2%}")
            st.write("Prediction (0 = not due, 1 = due):", pred)

    st.write("Model trained on synthetic data. Replace with real maintenance records for production use.")

# =========================================
# TAB 3: INVENTORY OPTIMIZATION (Step 2)
# =========================================
with tab3:
    st.title("📦 Inventory Optimization Module")

    st.write("""
    This module calculates:
    - **Forecasted consumption**
    - **Safety stock**
    - **Reorder point (ROP)**
    based on demand and supplier lead times.
    """)

    st.markdown("---")

    # User input for forecast horizon
    forecast_days = st.slider("Forecast Horizon (days)", 7, 120, 30)

    # For demo: estimated daily demand = failure_rate × usage scaling
    df["daily_demand"] = df["failure_rate_est"] * (df["operating_hours"] / 365)

    df["forecast_consumption"] = df["daily_demand"] * forecast_days

    # Safety stock formula
    Z = 1.65  # 95% service level
    df["demand_std"] = df["daily_demand"] * 0.3  # synthetic variability
    df["safety_stock"] = Z * df["demand_std"] * np.sqrt(df["lead_time_days"])

    # Reorder Point
    df["reorder_point"] = (df["daily_demand"] * df["lead_time_days"]) + df["safety_stock"]

    st.subheader("Inventory Optimization Result")
    st.dataframe(df[[
        "part_number", "machine_type", "part_category",
        "daily_demand", "forecast_consumption",
        "safety_stock", "reorder_point"
    ]].round(2))

    # Visuals
    st.subheader("Reorder Point by Part")
    fig4 = px.bar(df, x="part_number", y="reorder_point", color="part_category")
    st.plotly_chart(fig4, width='stretch')

# =========================================
# TAB 4: COST SIMULATION
# =========================================
with tab4:
    st.title("💰 Cost Simulation Module")

    st.write("""
    Simulate the financial impact of parts failure and downtime.
    Adjust machine costs, forecast horizon, and lead time multipliers to see effects.
    """)

    # User inputs
    forecast_days = st.slider("Forecast Horizon (days)", 7, 120, 30, key="cost_forecast_days")
    lead_time_multiplier = st.slider("Lead Time Multiplier", 1.0, 3.0, 1.0, key = "cost_lead_time_multiplier")
    
    st.subheader("Hourly Machine Cost by Machine Type (USD/hour)")
    machine_types = df["machine_type"].unique()
    machine_costs = {}
    for m in machine_types:
        machine_costs[m] = st.number_input(f"{m}", min_value=0.0, value=50.0)

    # Calculate daily demand (from Inventory module)
    df["daily_demand"] = df["failure_rate_est"] * (df["operating_hours"] / 365)
    df["forecast_consumption"] = df["daily_demand"] * forecast_days

    # Replacement cost
    df["replacement_cost"] = df["part_price_usd"] * df["forecast_consumption"]

    # Downtime cost
    df["downtime_cost"] = df.apply(lambda row: row["downtime_impact_hours"] * machine_costs[row["machine_type"]], axis=1)

    # Total risk cost
    df["total_risk_cost"] = (df["downtime_cost"] + df["replacement_cost"]) * df["failure_rate_est"] * lead_time_multiplier

    # Top 15 most costly parts
    top_costs = df.sort_values("total_risk_cost", ascending=False).head(15)

    st.subheader("Top 15 Parts by Total Risk Cost")
    st.dataframe(top_costs[[
        "part_number", "machine_type", "part_category",
        "failure_rate_est", "forecast_consumption",
        "downtime_cost", "replacement_cost", "total_risk_cost"
    ]].round(2))

    # Visualization: stacked bar
    st.subheader("Risk Cost Breakdown (Downtime vs Replacement)")
    import plotly.express as px
    fig_cost = px.bar(
        top_costs,
        x="part_number",
        y=["downtime_cost", "replacement_cost"],
        title="Downtime and Replacement Costs",
        labels={"value":"USD", "part_number":"Part Number"},
        color_discrete_sequence=["#EF553B","#636EFA"]
    )
    st.plotly_chart(fig_cost, use_container_width=True)

    # Optional: total risk cost summary
    st.subheader("Summary")
    total_downtime_cost = df["downtime_cost"].sum()
    total_replacement_cost = df["replacement_cost"].sum()
    total_risk_cost = df["total_risk_cost"].sum()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Downtime Cost", f"${total_downtime_cost:,.2f}")
    col2.metric("Total Replacement Cost", f"${total_replacement_cost:,.2f}")
    col3.metric("Total Risk Cost", f"${total_risk_cost:,.2f}")
