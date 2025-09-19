# app/streamlit_app.py
# Customer Behavior Intelligence — Interactive Dashboard
# Author: Capstone Project
# This app provides: data upload, cluster visualization, classification, regression (spending prediction).
# Comments are in English, all app text is concise and professional.

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

sns.set(style="whitegrid", palette="muted")

# Config
MODELS_DIR = Path("../models")
EXPECTED_FEATURES = ["feature_0", "feature_1", "feature_2", "feature_3", "feature_4"]

# -------- Load artifacts --------
@st.cache_resource
def load_artifacts():
    artifacts = {}
    p_clf = MODELS_DIR / "classifier.joblib"
    p_reg_lin = MODELS_DIR / "lin_reg.joblib"
    p_reg_gb = MODELS_DIR / "gb_reg.joblib"

    artifacts["classifier"] = joblib.load(p_clf) if p_clf.exists() else None
    artifacts["regressor_lin"] = joblib.load(p_reg_lin) if p_reg_lin.exists() else None
    artifacts["regressor_gb"] = joblib.load(p_reg_gb) if p_reg_gb.exists() else None
    return artifacts

def preprocess_df_for_model(df):
    return df[EXPECTED_FEATURES].values

def pca_2d_plot(X, labels=None, title="PCA projection"):
    pca = PCA(n_components=2, random_state=42)
    proj = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(7,5))
    if labels is not None:
        unique = np.unique(labels)
        palette = sns.color_palette("Set2", n_colors=len(unique))
        for idx, u in enumerate(unique):
            mask = labels == u
            ax.scatter(proj[mask,0], proj[mask,1], s=50, alpha=0.75, label=str(u), color=palette[idx])
        ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1))
    else:
        ax.scatter(proj[:,0], proj[:,1], s=40, alpha=0.7)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    plt.tight_layout()
    return fig

def df_to_download_bytes(df, filename="predictions.csv"):
    towrite = BytesIO()
    df.to_csv(towrite, index=False)
    towrite.seek(0)
    return towrite, filename

# -------- App UI --------
st.set_page_config(page_title="CBI Dashboard", layout="wide")
st.title("Customer Behavior Intelligence — Dashboard")

artifacts = load_artifacts()
classifier = artifacts["classifier"]
reg_lin = artifacts["regressor_lin"]
reg_gb = artifacts["regressor_gb"]

# Sidebar: file upload
with st.sidebar:
    st.header("Controls")
    uploaded = st.file_uploader("Upload cleaned CSV", type=["csv"])
    st.write("Artifacts loaded:")
    st.write(f"Classifier: {'✅' if classifier else '❌'}")
    st.write(f"Linear Regressor: {'✅' if reg_lin else '❌'}")
    st.write(f"GB Regressor: {'✅' if reg_gb else '❌'}")

# Load dataframe
df = None
if uploaded:
    try:
        df = pd.read_csv(uploaded)
        st.success(f"Loaded {df.shape[0]} rows x {df.shape[1]} cols")
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")

if df is None:
    st.info("Upload a cleaned CSV with required features to proceed.")
    st.stop()

# Data preview
st.subheader("Data preview")
st.dataframe(df.head(8))

# Predict cluster if not present
if "cluster" not in df.columns and classifier is not None:
    X_proc = preprocess_df_for_model(df)
    preds = classifier.predict(X_proc)
    df["cluster"] = preds
    st.info("Cluster column added using classifier.")

# PCA visualization
st.subheader("Cluster visualization (PCA)")
X_proc = preprocess_df_for_model(df)
fig = pca_2d_plot(X_proc, labels=df["cluster"].values if "cluster" in df.columns else None)
st.pyplot(fig)

# Feature importances
st.subheader("Model insights")
cols = st.columns(2)
with cols[0]:
    if classifier is not None and hasattr(classifier, "feature_importances_"):
        fi = pd.Series(classifier.feature_importances_, index=EXPECTED_FEATURES).sort_values(ascending=False)
        st.bar_chart(fi)
        st.write(fi)
with cols[1]:
    if reg_lin is not None and hasattr(reg_lin, "coef_"):
        coefs = pd.Series(reg_lin.coef_.flatten(), index=EXPECTED_FEATURES).sort_values(key=abs, ascending=False)
        st.bar_chart(coefs)
        st.write(coefs)
    elif reg_gb is not None and hasattr(reg_gb, "feature_importances_"):
        rif = pd.Series(reg_gb.feature_importances_, index=EXPECTED_FEATURES).sort_values(ascending=False)
        st.bar_chart(rif)
        st.write(rif)

# Single user prediction
st.subheader("Single user prediction")
vals = {}
for f in EXPECTED_FEATURES:
    vals[f] = st.number_input(f, value=float(df[f].median()), format="%.4f")
single_df = pd.DataFrame([vals])
Xs = preprocess_df_for_model(single_df)
if classifier:
    cluster_pred = int(classifier.predict(Xs)[0])
    st.metric("Predicted cluster", cluster_pred)
if reg_lin:
    spend_lin = float(reg_lin.predict(Xs)[0])
    st.metric("Predicted spending (Linear)", f"{spend_lin:.2f}")
if reg_gb:
    spend_gb = float(reg_gb.predict(Xs)[0])
    st.metric("Predicted spending (GB)", f"{spend_gb:.2f}")

# Batch prediction
st.subheader("Batch prediction")
batch_file = st.file_uploader("Upload batch CSV", type=["csv"], key="batch")
if batch_file:
    bdf = pd.read_csv(batch_file)
    Xb = preprocess_df_for_model(bdf)
    if classifier:
        bdf["cluster"] = classifier.predict(Xb)
    if reg_lin:
        bdf["predicted_spending_lin"] = reg_lin.predict(Xb)
    if reg_gb:
        bdf["predicted_spending_gb"] = reg_gb.predict(Xb)
    st.dataframe(bdf.head(8))
    towrite, fname = df_to_download_bytes(bdf)
    st.download_button("Download predictions CSV", towrite, file_name=fname)

st.markdown("---")
st.caption("CBI Dashboard — Clustering → Classification → Regression")

# -------- Business-Friendly Summary --------
st.subheader("Business-Friendly Summary of Findings")

# 1 Cluster Insights
st.markdown("""
**Cluster Insights:**
- **Cluster 0 — Power Users**  
  High engagement on feature_0, above-average on feature_1. Likely highly active and loyal users.
- **Cluster 1 — Low-activity but Paying Users**  
  Moderate on feature_0, low on feature_2 & feature_3, above-average on feature_4. Limited activity but higher spending potential.
- **Cluster 2 — Specialized High Spenders**  
  Low on feature_0 & feature_1, strong activity on feature_2, very high on feature_4. Specialized users with high spending on specific services.
- **Cluster 3 — Low-engagement Users**  
  Very low on feature_0, above-average on feature_1, low on feature_3. Users with low engagement, may need reactivation campaigns.
""")

# 2 Business Recommendations
st.markdown("""
**Business Recommendations:**
- **Cluster 0:** Reward loyalty → offer exclusive benefits.
- **Cluster 1 & 2:** Focus on upselling and premium offers → higher spending potential.
- **Cluster 3:** Launch reactivation campaigns → targeted marketing to boost engagement.
""")

# 3 Classification Notes
st.markdown("""
**Classification Insights:**
- Classification achieved 100% accuracy (RandomForest, Logistic Regression, SVM) due to highly separable synthetic clusters.
- Overfitting handled with label noise injection → realistic accuracy dropped to 91%.
- Interpretation: New users can be assigned to clusters with high confidence based on behavioral features.
""")

# 4 Regression / Spending Prediction
st.markdown(f"""
**Spending Prediction Results:**
- **Linear Regression:** R²=0.971 | MAE≈7 | RMSE≈9 → best performance
- **Gradient Boosting:** R²=0.943 | MAE≈9.75 | RMSE≈13.19
- **Top drivers of spending:**  
  - feature_4 (~54% contribution)  
  - feature_1 (~36% contribution)  
  - feature_0 & feature_2 minor influence  
  - feature_3 negligible
- **Business Interpretation:**  
  Users with higher feature_4 & feature_1 scores are most valuable. Focus premium strategies on these clusters.
""")

