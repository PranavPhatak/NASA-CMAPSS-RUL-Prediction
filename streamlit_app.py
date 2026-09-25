import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("CMAPSS_API_URL", "http://localhost:8000")

FEATURE_COLUMNS = [
    "setting_1", "setting_2", "setting_3",
    "sensor_2", "sensor_3", "sensor_4", "sensor_6", "sensor_7", "sensor_8",
    "sensor_9", "sensor_11", "sensor_12", "sensor_13", "sensor_14",
    "sensor_15", "sensor_17", "sensor_20", "sensor_21",
]

st.set_page_config(page_title="CMAPSS RUL Predictor", page_icon="🛩️", layout="centered")

st.title("🛩️ Turbofan Engine RUL Predictor")
st.caption("NASA C-MAPSS FD001 · Bidirectional LSTM · Reduced sensors, RUL capped at 125 cycles")

with st.sidebar:
    st.subheader("API status")
    try:
        health = requests.get(f"{API_URL}/health", timeout=5).json()
        if health["status"] == "ok":
            st.success("Connected — model & scaler loaded")
        else:
            st.error(f"API error: {health.get('error')}")
        st.write(f"Window size: **{health['window_size']}** cycles")
        st.write(f"Features expected: **{health['n_features']}**")
    except Exception:
        st.error(f"Cannot reach API at {API_URL}\nStart it with:\n`uvicorn api:app --reload`")

    st.divider()
    st.caption("API URL")
    st.code(API_URL, language=None)

tab_csv, tab_manual = st.tabs(["📄 Upload CSV", "✍️ Manual entry"])

with tab_csv:
    st.write(
        "Upload a CSV containing **one engine's** cycles, sorted oldest → newest. "
        "`unit_id` / `cycle` / `RUL` columns are fine to include — they're ignored."
    )
    uploaded = st.file_uploader("Engine cycle CSV", type=["csv"])

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.write(f"Loaded **{len(df)}** rows.")

        if "unit_id" in df.columns and df["unit_id"].nunique() > 1:
            engine_ids = sorted(df["unit_id"].unique())
            chosen = st.selectbox("Multiple engines detected — pick one:", engine_ids)
            df = df[df["unit_id"] == chosen]

        if "cycle" in df.columns:
            df = df.sort_values("cycle")

        st.dataframe(df.tail(10), use_container_width=True)

        missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
        if missing:
            st.error(f"CSV is missing required columns: {missing}")
        elif st.button("Predict RUL", type="primary", key="predict_csv"):
            rows = df[FEATURE_COLUMNS].to_dict(orient="records")
            with st.spinner("Running inference..."):
                try:
                    resp = requests.post(f"{API_URL}/predict_csv", json=rows, timeout=30)
                    resp.raise_for_status()
                    result = resp.json()
                    st.metric("Predicted Remaining Useful Life", f"{result['predicted_rul']:.1f} cycles")
                    st.caption(
                        f"Used last {result['cycles_used']} of {result['cycles_received']} cycles"
                        + (" (padded — fewer than 30 available)" if result["was_padded"] else "")
                    )
                    if result["predicted_rul"] < 20:
                        st.error("⚠️ Engine is approaching failure — schedule maintenance soon.")
                    elif result["predicted_rul"] < 50:
                        st.warning("Engine health is declining — monitor closely.")
                    else:
                        st.success("Engine looks healthy.")
                except requests.exceptions.RequestException as e:
                    st.error(f"Request failed: {e}")

with tab_manual:
    st.write(
        "Enter one cycle's readings — it will be repeated to fill the model's "
        "30-cycle window (a rough single-snapshot estimate)."
    )

    defaults = {
        "setting_1": 0.0, "setting_2": 0.0, "setting_3": 100.0,
        "sensor_2": 642.0, "sensor_3": 1590.0, "sensor_4": 1400.0,
        "sensor_6": 21.6, "sensor_7": 554.0, "sensor_8": 2388.0,
        "sensor_9": 9050.0, "sensor_11": 47.3, "sensor_12": 521.5,
        "sensor_13": 2388.0, "sensor_14": 8130.0, "sensor_15": 8.42,
        "sensor_17": 393.0, "sensor_20": 39.0, "sensor_21": 23.4,
    }

    cols = st.columns(3)
    values = {}
    for i, feat in enumerate(FEATURE_COLUMNS):
        with cols[i % 3]:
            values[feat] = st.number_input(feat, value=float(defaults[feat]), format="%.4f")

    if st.button("Predict RUL", type="primary", key="predict_manual"):
        payload = {"cycles": [values]}
        with st.spinner("Running inference..."):
            try:
                resp = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
                resp.raise_for_status()
                result = resp.json()
                st.metric("Predicted Remaining Useful Life", f"{result['predicted_rul']:.1f} cycles")
                st.caption("Single cycle repeated to fill the 30-cycle window.")
            except requests.exceptions.RequestException as e:
                st.error(f"Request failed: {e}")

st.divider()
st.caption(
    "Model: Bidirectional LSTM trained on FD001 with 6 constant sensors removed "
    "(18 features), 30-cycle windows, RUL capped at 125."
)
