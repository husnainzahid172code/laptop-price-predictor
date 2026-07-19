import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import altair as alt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

st.set_page_config(page_title="Laptop Price Predictor", layout="wide")
st.title("Laptop Price Prediction System")
st.markdown("Predict the price of a laptop based on its specifications using machine learning.")

@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(MODEL_DIR, "laptop_price_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    encoders = joblib.load(os.path.join(MODEL_DIR, "encoders.pkl"))
    feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_cols.pkl"))
    results_df = joblib.load(os.path.join(MODEL_DIR, "results.pkl"))
    return model, scaler, encoders, feature_cols, results_df

@st.cache_data
def load_data():
    return pd.read_csv(os.path.join(BASE_DIR, "laptops_clean.csv"))

if not os.path.exists(os.path.join(MODEL_DIR, "laptop_price_model.pkl")):
    st.warning("Model not found! Please run `python train_model.py` first.")
    st.stop()

df = load_data()
model, scaler, encoders, feature_cols, results_df = load_artifacts()

numeric_cols = ['Screen_Size_Inches', 'CPU_Speed_GHz', 'RAM_GB', 'Storage_GB', 'Weight_KG']
cat_cols = ['Company', 'Type', 'CPU_Brand', 'Storage_Type', 'GPU_Brand']
has_hybrid = 'Is_Hybrid_Storage' in feature_cols

tab1, tab2, tab3 = st.tabs(["Predict Price", "Model Performance", "EDA & Insights"])

with tab1:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Laptop Specifications")
        company = st.selectbox("Company", sorted(df['Company'].unique()))
        laptop_type = st.selectbox("Type", sorted(df['Type'].unique()))
        cpu_brand = st.selectbox("CPU Brand", sorted(df['CPU_Brand'].unique()))
        cpu_speed = st.slider("CPU Speed (GHz)", float(df['CPU_Speed_GHz'].min()), float(df['CPU_Speed_GHz'].max()), 2.0, 0.1)
        ram = st.selectbox("RAM (GB)", sorted(df['RAM_GB'].unique()))
        storage_gb = st.selectbox("Storage (GB)", sorted(df['Storage_GB'].unique()))
        storage_type = st.selectbox("Storage Type", sorted(df['Storage_Type'].unique()))
        gpu_brand = st.selectbox("GPU Brand", sorted(df['GPU_Brand'].unique()))
        screen_size = st.slider("Screen Size (inches)", float(df['Screen_Size_Inches'].min()), float(df['Screen_Size_Inches'].max()), 15.6, 0.1)
        touchscreen = st.checkbox("Touchscreen")
        ips_panel = st.checkbox("IPS Panel")
        weight = st.slider("Weight (kg)", float(df['Weight_KG'].min()), float(df['Weight_KG'].max()), 2.0, 0.01)

    with col2:
        st.subheader("Predicted Price")
        if st.button("Predict Price", type="primary", use_container_width=True):
            input_data = pd.DataFrame([{
                'Company': company, 'Type': laptop_type,
                'Screen_Size_Inches': screen_size, 'Touchscreen': int(touchscreen),
                'IPS_Panel': int(ips_panel), 'CPU_Brand': cpu_brand,
                'CPU_Speed_GHz': cpu_speed, 'RAM_GB': ram, 'Storage_GB': storage_gb,
                'Storage_Type': storage_type, 'GPU_Brand': gpu_brand, 'Weight_KG': weight
            }])
            if has_hybrid:
                input_data['Is_Hybrid_Storage'] = 0
            for c in cat_cols:
                input_data[c] = encoders[c].transform(input_data[c].astype(str))
            input_scaled = input_data.copy()
            inp_numeric = [c for c in numeric_cols if c in input_scaled.columns]
            input_scaled[inp_numeric] = scaler.transform(input_scaled[inp_numeric])

            prediction = model.predict(input_scaled)[0]
            st.metric("Estimated Price", f"€{prediction:,.2f}")

            similar = df[
                (df['RAM_GB'] == ram) &
                (df['Storage_Type'] == storage_type) &
                (df['Company'] == company)
            ]['Price_Euros']
            if len(similar) > 0:
                st.caption(f"Similar laptops range: €{similar.min():,.0f} - €{similar.max():,.0f}")

            st.subheader("Specs Summary")
            specs = [
                f"**{company}** | {laptop_type}",
                f"{cpu_brand} @ {cpu_speed}GHz | {ram}GB RAM",
                f"{storage_gb}GB {storage_type}",
                f"{screen_size}\" {'Touch' if touchscreen else ''} {'IPS' if ips_panel else ''}",
                f"{weight}kg | {gpu_brand} GPU"
            ]
            for s in specs:
                st.write(s)

with tab2:
    st.subheader("Model Performance Comparison")
    st.dataframe(results_df.style.format("{:.4f}"), use_container_width=True)

    X_all = df[[c for c in feature_cols if c in df.columns]].copy()
    for c in cat_cols:
        X_all[c] = encoders[c].transform(X_all[c].astype(str))
    if has_hybrid and 'Is_Hybrid_Storage' in X_all.columns:
        X_all['Is_Hybrid_Storage'] = X_all['Is_Hybrid_Storage'].astype(int)
    inp_numeric = [c for c in numeric_cols if c in X_all.columns]
    X_all[inp_numeric] = scaler.transform(X_all[inp_numeric])
    y_all_true = df['Price_Euros'].values
    y_all_pred = model.predict(X_all)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Actual vs Predicted Price")
        pred_df = pd.DataFrame({"Actual": y_all_true, "Predicted": y_all_pred})
        chart = alt.Chart(pred_df).mark_circle(opacity=0.4, size=30).encode(
            x=alt.X("Actual:Q", title="Actual Price (€)"),
            y=alt.Y("Predicted:Q", title="Predicted Price (€)")
        ).properties(height=400)
        line = alt.Chart(pred_df).mark_line(color="red", strokeDash=[4, 4]).encode(
            x=alt.X("Actual:Q"),
            y=alt.Y("Actual:Q")
        )
        st.altair_chart(chart + line, use_container_width=True)

    with col_b:
        if hasattr(model, 'feature_importances_'):
            st.subheader("Feature Importance")
            imp_df = pd.DataFrame({
                'feature': feature_cols,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=True)
            imp_chart = alt.Chart(imp_df).mark_bar(color="steelblue").encode(
                x=alt.X("importance:Q", title="Importance"),
                y=alt.Y("feature:N", title="", sort=None)
            ).properties(height=400)
            st.altair_chart(imp_chart, use_container_width=True)

with tab3:
    st.subheader("Data Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Laptops", len(df))
    c2.metric("Avg Price", f"€{df['Price_Euros'].mean():,.0f}")
    c3.metric("Avg RAM", f"{df['RAM_GB'].mean():.1f} GB")
    c4.metric("Avg Storage", f"{df['Storage_GB'].mean():,.0f} GB")

    chart_type = st.selectbox("Select Visualization",
        ["Price Distribution", "Avg Price by Company", "RAM vs Price",
         "Price by Type", "Correlation Heatmap", "Storage Type vs Price"])

    if chart_type == "Price Distribution":
        hist = alt.Chart(df).mark_bar(opacity=0.7).encode(
            alt.X("Price_Euros:Q", bin=alt.Bin(maxbins=40), title="Price (Euros)"),
            alt.Y("count()", title="Count")
        ).properties(height=400)
        st.altair_chart(hist, use_container_width=True)

    elif chart_type == "Avg Price by Company":
        avg_df = df.groupby('Company')['Price_Euros'].mean().reset_index().sort_values('Price_Euros', ascending=False)
        bar = alt.Chart(avg_df).mark_bar().encode(
            x=alt.X("Company:N", sort=None),
            y=alt.Y("Price_Euros:Q", title="Avg Price (€)"),
            color=alt.Color("Company:N", legend=None)
        ).properties(height=400)
        st.altair_chart(bar, use_container_width=True)

    elif chart_type == "RAM vs Price":
        scatter = alt.Chart(df).mark_circle(opacity=0.6, size=40).encode(
            x=alt.X("RAM_GB:Q", title="RAM (GB)"),
            y=alt.Y("Price_Euros:Q", title="Price (€)"),
            color=alt.Color("Type:N", legend=alt.Legend(orient="right"))
        ).properties(height=400)
        st.altair_chart(scatter, use_container_width=True)

    elif chart_type == "Price by Type":
        box = alt.Chart(df).mark_boxplot().encode(
            x=alt.X("Type:N", sort=alt.EncodingSortField(field="Price_Euros", op="median", order="descending")),
            y=alt.Y("Price_Euros:Q", title="Price (€)")
        ).properties(height=400)
        st.altair_chart(box, use_container_width=True)

    elif chart_type == "Correlation Heatmap":
        corr = df.select_dtypes(include=[np.number]).corr().reset_index().melt(id_vars="index")
        heat = alt.Chart(corr).mark_rect().encode(
            x=alt.X("index:N", title=""),
            y=alt.Y("variable:N", title=""),
            color=alt.Color("value:Q", scale=alt.Scale(scheme="coolwarm"), title="Correlation")
        ).properties(height=400)
        text = heat.mark_text().encode(text=alt.Text("value:Q", format=".2f"))
        st.altair_chart(heat + text, use_container_width=True)

    elif chart_type == "Storage Type vs Price":
        stor_df = df.groupby('Storage_Type')['Price_Euros'].mean().reset_index()
        bar = alt.Chart(stor_df).mark_bar().encode(
            x=alt.X("Storage_Type:N", sort=None),
            y=alt.Y("Price_Euros:Q", title="Avg Price (€)"),
            color=alt.Color("Storage_Type:N", legend=None)
        ).properties(height=400)
        st.altair_chart(bar, use_container_width=True)
