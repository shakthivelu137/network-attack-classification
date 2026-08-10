"""
Network Attack Classification System - Interactive Dashboard
Run with: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Network Attack Classification System", layout="wide", page_icon="🛡️")

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

@st.cache_resource
def load_artifacts():
    rf = joblib.load(f"{MODEL_DIR}/random_forest.pkl")
    dt = joblib.load(f"{MODEL_DIR}/decision_tree.pkl")
    svm = joblib.load(f"{MODEL_DIR}/svm.pkl")
    scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")
    encoders = joblib.load(f"{MODEL_DIR}/encoders.pkl")
    target_le = joblib.load(f"{MODEL_DIR}/target_encoder.pkl")
    feature_cols = joblib.load(f"{MODEL_DIR}/feature_cols.pkl")
    X_val = joblib.load(f"{MODEL_DIR}/X_val_sample.pkl")
    y_val = joblib.load(f"{MODEL_DIR}/y_val_sample.pkl")
    with open(f"{MODEL_DIR}/results.json") as f:
        results = json.load(f)
    with open(f"{MODEL_DIR}/confusion_matrix.json") as f:
        cm_data = json.load(f)
    with open(f"{MODEL_DIR}/feature_importance.json") as f:
        importances = json.load(f)
    return rf, dt, svm, scaler, encoders, target_le, feature_cols, X_val, y_val, results, cm_data, importances

rf, dt, svm, scaler, encoders, target_le, feature_cols, X_val, y_val, results, cm_data, importances = load_artifacts()
MODELS = {"Random Forest": rf, "Decision Tree": dt, "SVM": svm}

CATEGORY_INFO = {
    "normal": {"color": "#2ecc71", "desc": "Legitimate, non-malicious network traffic."},
    "dos": {"color": "#e74c3c", "desc": "Denial of Service — floods/overwhelms a target to deny service."},
    "probe": {"color": "#f39c12", "desc": "Surveillance/scanning to gather information about a network."},
    "r2l": {"color": "#9b59b6", "desc": "Remote to Local — unauthorized access from a remote machine."},
    "u2r": {"color": "#34495e", "desc": "User to Root — unauthorized access to root/admin privileges."},
}

st.title("🛡️ Network Attack Classification System")
st.caption("ML-based intrusion detection on the NSL-KDD dataset · Decision Tree · Random Forest · SVM")

tab1, tab2, tab3, tab4 = st.tabs(["🔍 Live Classifier", "📊 Model Performance", "🧩 Feature Importance", "ℹ️ About the Project"])

# ---------------- TAB 1: Live Classifier ----------------
with tab1:
    st.subheader("Classify a Network Connection")
    st.write("Pick a sample from the validation set, or adjust values manually, then classify it.")

    col_a, col_b = st.columns([1, 2])
    with col_a:
        model_choice = st.selectbox("Model", list(MODELS.keys()), index=0)
        sample_idx = st.number_input("Sample index (0 - %d)" % (len(X_val) - 1), min_value=0, max_value=len(X_val)-1, value=0, step=1)
        if st.button("🎲 Random sample"):
            sample_idx = int(np.random.randint(0, len(X_val)))
            st.session_state["sample_idx"] = sample_idx
        if "sample_idx" in st.session_state:
            sample_idx = st.session_state["sample_idx"]

    row = X_val.iloc[[sample_idx]]
    true_label = target_le.inverse_transform([y_val[sample_idx]])[0]

    with col_b:
        st.write("**Selected connection record (key fields):**")
        display_cols = ["protocol_type", "service", "flag", "src_bytes", "dst_bytes", "count", "srv_count", "logged_in"]
        show = row[display_cols].copy()
        show["protocol_type"] = encoders["protocol_type"].inverse_transform(show["protocol_type"])
        show["service"] = encoders["service"].inverse_transform(show["service"])
        show["flag"] = encoders["flag"].inverse_transform(show["flag"])
        st.dataframe(show, hide_index=True, use_container_width=True)

    if st.button("🚀 Classify", type="primary"):
        model = MODELS[model_choice]
        X_input = row[feature_cols]
        if model_choice == "SVM":
            X_input_used = scaler.transform(X_input)
        else:
            X_input_used = X_input
        pred = model.predict(X_input_used)[0]
        pred_label = target_le.inverse_transform([pred])[0]

        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_input_used)[0]

        info = CATEGORY_INFO.get(pred_label, {})
        correct = pred_label == true_label

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"### Prediction: `{pred_label.upper()}`")
            st.write(info.get("desc", ""))
            if correct:
                st.success(f"✅ Matches ground truth ({true_label})")
            else:
                st.error(f"⚠️ Ground truth was `{true_label}`")
        with c2:
            if proba is not None:
                proba_df = pd.DataFrame({
                    "category": target_le.classes_,
                    "probability": proba
                }).sort_values("probability", ascending=False)
                fig = px.bar(proba_df, x="probability", y="category", orientation="h",
                             color="category", color_discrete_map={k: v["color"] for k, v in CATEGORY_INFO.items()})
                fig.update_layout(showlegend=False, height=280, margin=dict(t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)

# ---------------- TAB 2: Model Performance ----------------
with tab2:
    st.subheader("Model Comparison")
    st.write("**Validation split (80/20, same distribution as training)** — the standard evaluation method used in most NSL-KDD literature:")
    val_df = pd.DataFrame(results["validation_split"]).T
    st.dataframe(val_df.style.format("{:.2%}").background_gradient(cmap="Greens"), use_container_width=True)

    st.write("**KDDTest+ (contains genuinely unseen attack types)** — a harder, more realistic generalization test:")
    test_df = pd.DataFrame(results["kddtest_unseen_attacks"]).T
    st.dataframe(test_df.style.format("{:.2%}").background_gradient(cmap="Oranges"), use_container_width=True)

    st.info(
        "💡 **Why two tables?** NSL-KDD's official test set deliberately includes attack variants not seen "
        "during training, so accuracy naturally drops there. The ~99% figure (validation split) is the number "
        "most commonly cited in papers/projects; the KDDTest+ figure reflects true generalization to novel attacks."
    )

    st.subheader("Confusion Matrix — Random Forest (validation split)")
    cm = np.array(cm_data["matrix"])
    labels = cm_data["labels"]
    fig_cm = go.Figure(data=go.Heatmap(
        z=cm, x=labels, y=labels, colorscale="Blues", text=cm, texttemplate="%{text}"
    ))
    fig_cm.update_layout(xaxis_title="Predicted", yaxis_title="Actual", height=450)
    st.plotly_chart(fig_cm, use_container_width=True)

# ---------------- TAB 3: Feature Importance ----------------
with tab3:
    st.subheader("Which features matter most? (Random Forest)")
    imp_df = pd.DataFrame(list(importances.items()), columns=["feature", "importance"]).sort_values("importance", ascending=True).tail(15)
    fig_imp = px.bar(imp_df, x="importance", y="feature", orientation="h", color="importance", color_continuous_scale="Viridis")
    fig_imp.update_layout(height=550, coloraxis_showscale=False)
    st.plotly_chart(fig_imp, use_container_width=True)
    st.caption("Top predictors typically include service type, error rates, and connection counts — consistent with known DoS/probe signatures.")

# ---------------- TAB 4: About ----------------
with tab4:
    st.subheader("About this project")
    st.markdown("""
**Network Attack Classification System**

A machine learning-based intrusion detection system trained on the **NSL-KDD** dataset
(an improved version of the classic KDD Cup 1999 dataset), classifying network connections
into 5 categories:

- **Normal** — legitimate traffic
- **DoS** — Denial of Service attacks
- **Probe** — surveillance/scanning
- **R2L** — Remote-to-Local unauthorized access
- **U2R** — User-to-Root privilege escalation

**Pipeline:**
1. Loaded 125,973 training records / 22,544 test records from NSL-KDD
2. Label-encoded categorical features (protocol, service, flag)
3. Standard-scaled numeric features for SVM
4. Trained Decision Tree, Random Forest, and SVM classifiers
5. Evaluated with accuracy, precision, recall, F1, and confusion matrix

**Tech stack:** Python, scikit-learn, pandas, Streamlit, Plotly

**Best model:** Random Forest — 99.9% accuracy (validation split), 75% on unseen-attack test set
    """)
