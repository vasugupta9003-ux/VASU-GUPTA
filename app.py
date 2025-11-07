
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, RocCurveDisplay
)

@st.cache_data
def load_data(csv_path: str = "UniversalBank.csv"):
    return pd.read_csv(csv_path)

def get_features_target(df: pd.DataFrame):
    X = df.drop(columns=["ID", "ZIP Code", "Personal Loan"])
    y = df["Personal Loan"]
    return X, y

def train_all_models(df: pd.DataFrame):
    X, y = get_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }

    rows = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        model.fit(X_train, y_train)

        y_tr = model.predict(X_train)
        y_te = model.predict(X_test)
        proba_te = model.predict_proba(X_test)[:, 1]

        train_acc = accuracy_score(y_train, y_tr)
        test_acc = accuracy_score(y_test, y_te)
        precision = precision_score(y_test, y_te)
        recall = recall_score(y_test, y_te)
        f1 = f1_score(y_test, y_te)
        auc = roc_auc_score(y_test, proba_te)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")

        rows.append({
            "Model": name,
            "Train Accuracy": round(train_acc, 4),
            "Test Accuracy": round(test_acc, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1-Score": round(f1, 4),
            "AUC": round(auc, 4),
            "CV Accuracy (5-fold)": round(cv_scores.mean(), 4),
        })

    metrics_df = pd.DataFrame(rows)
    return {
        "models": models,
        "metrics": metrics_df,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }

def customer_insights_tab(df: pd.DataFrame):
    st.subheader("Customer Insights for Better Personal Loan Conversions")

    c1, c2, c3 = st.columns(3)
    with c1:
        edu = st.multiselect(
            "Filter by Education",
            options=sorted(df["Education"].unique()),
            default=sorted(df["Education"].unique()),
        )
    with c2:
        online = st.multiselect(
            "Filter by Online Banking (0=No,1=Yes)",
            options=sorted(df["Online"].unique()),
            default=sorted(df["Online"].unique()),
        )
    with c3:
        cd = st.multiselect(
            "Filter by CD Account (0=No,1=Yes)",
            options=sorted(df["CD Account"].unique()),
            default=sorted(df["CD Account"].unique()),
        )

    fdf = df[
        (df["Education"].isin(edu))
        & (df["Online"].isin(online))
        & (df["CD Account"].isin(cd))
    ]

    # 1) Income vs Education vs Personal Loan
    st.markdown("### 1️⃣ Income vs Education vs Loan (Segmented Boxplot)")
    fig1 = px.box(
        fdf,
        x="Education",
        y="Income",
        color="Personal Loan",
        points="outliers",
        labels={"Education": "Education Level", "Income": "Annual Income ($000)"},
    )
    st.plotly_chart(fig1, use_container_width=True)
    st.caption("- High-income Graduate & Advanced customers are prime loan targets.")

    # 2) Credit Card Spending vs Loan
    st.markdown("### 2️⃣ Credit Card Spending vs Loan Acceptance")
    fig2 = px.violin(
        fdf,
        x="Personal Loan",
        y="CCAvg",
        box=True,
        points="all",
        labels={"CCAvg": "Avg Monthly Card Spend ($000)"},
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("- Heavy card users show higher inclination towards personal loans.")

    # 3) Family Size & Online Usage
    st.markdown("### 3️⃣ Family Size & Digital Usage vs Loan")
    fdf["Family_Str"] = fdf["Family"].astype(str)
    fig3 = px.histogram(
        fdf,
        x="Family_Str",
        color="Personal Loan",
        barmode="group",
        facet_col="Online",
        labels={"Family_Str": "Family Size"},
    )
    st.plotly_chart(fig3, use_container_width=True)
    st.caption("- Families of 3–4 using online banking are attractive segments.")

    # 4) Product Holding Heatmap
    st.markdown("### 4️⃣ Product Holding Heatmap (CD, Securities, Online)")
    pivot = (
        fdf.groupby(["CD Account", "Securities Account", "Online"])["Personal Loan"]
        .mean()
        .reset_index()
    )
    pivot["Conversion_Rate"] = (pivot["Personal Loan"] * 100).round(1)
    fig4 = px.density_heatmap(
        pivot,
        x="CD Account",
        y="Online",
        z="Conversion_Rate",
        facet_col="Securities Account",
        text_auto=True,
        labels={"Conversion_Rate": "Loan Acceptance Rate (%)"},
    )
    st.plotly_chart(fig4, use_container_width=True)
    st.caption("- CD/Securities + Online users are high-conversion micro-clusters.")

    # 5) Correlation Heatmap
    st.markdown("### 5️⃣ Correlation Matrix of Key Drivers")
    cols = [
        "Age", "Experience", "Income", "Family",
        "CCAvg", "Education", "Mortgage", "Personal Loan",
    ]
    corr = fdf[cols].corr()
    fig5, ax5 = plt.subplots(figsize=(8, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", ax=ax5)
    ax5.set_title("Correlation Matrix")
    st.pyplot(fig5)
    st.caption("- Income, CCAvg and Education emerge as key loan drivers.")

def plot_roc_curves(models, X_test, y_test):
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, model in models.items():
        RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name=name)
    ax.set_title("ROC Curve Comparison (All Models)")
    st.pyplot(fig)

def plot_confusion_matrices(models, X_train, X_test, y_train, y_test):
    for name, model in models.items():
        st.markdown(f"#### {name} - Confusion Matrices")

        y_tr = model.predict(X_train)
        cm_tr = confusion_matrix(y_train, y_tr)
        fig_tr, ax_tr = plt.subplots()
        sns.heatmap(cm_tr, annot=True, fmt="d", ax=ax_tr)
        ax_tr.set_title(f"{name} - Train")
        ax_tr.set_xlabel("Predicted")
        ax_tr.set_ylabel("Actual")
        st.pyplot(fig_tr)

        y_te = model.predict(X_test)
        cm_te = confusion_matrix(y_test, y_te)
        fig_te, ax_te = plt.subplots()
        sns.heatmap(cm_te, annot=True, fmt="d", ax=ax_te)
        ax_te.set_title(f"{name} - Test")
        ax_te.set_xlabel("Predicted")
        ax_te.set_ylabel("Actual")
        st.pyplot(fig_te)

def plot_feature_importances(models, feature_names):
    for name, model in models.items():
        if not hasattr(model, "feature_importances_"):
            continue
        importances = model.feature_importances_
        idx = np.argsort(importances)[::-1]
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(x=importances[idx], y=np.array(feature_names)[idx], ax=ax)
        ax.set_title(f"{name} - Feature Importance")
        ax.set_xlabel("Importance Score")
        ax.set_ylabel("Feature")
        st.pyplot(fig)

def upload_and_predict_tab(base_df, stored_models):
    st.subheader("Upload New Customer Data & Predict Personal Loan")

    if stored_models is None:
        st.info("Run models first in the 'Model Performance' tab.")
        return

    model_names = list(stored_models.keys())
    selected = st.selectbox("Select Model for Prediction", model_names)
    model = stored_models[selected]

    file = st.file_uploader(
        "Upload CSV (similar structure to UniversalBank.csv)", type=["csv"]
    )

    if file is not None:
        new_df = pd.read_csv(file)
        original_df = new_df.copy()

        required = {
            "Age", "Experience", "Income", "Family", "CCAvg",
            "Education", "Mortgage", "Securities Account",
            "CD Account", "Online", "CreditCard",
        }
        missing = required - set(new_df.columns)
        if missing:
            st.error(f"Missing required columns: {missing}")
            return

        for col in ["ID", "ZIP Code", "Personal Loan"]:
            if col in new_df.columns:
                new_df = new_df.drop(columns=[col])

        X_new = new_df
        proba = model.predict_proba(X_new)[:, 1]
        pred = (proba >= 0.5).astype(int)

        result = original_df.copy()
        result["Predicted_Personal_Loan"] = pred
        result["Predicted_Probability"] = proba.round(4)

        st.write("Sample Predictions")
        st.dataframe(result.head())

        csv_data = result.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Predictions CSV",
            data=csv_data,
            file_name="personal_loan_predictions.csv",
            mime="text/csv",
        )
        st.success("Predictions generated successfully.")

def main():
    st.set_page_config(
        page_title="Universal Bank - Personal Loan Dashboard",
        layout="wide",
    )

    st.title("Universal Bank | Personal Loan Conversion Analytics")

    st.write(
        """Use this dashboard as **Marketing Head** to:
        - Discover high-probability personal loan segments.
        - Compare three ML algorithms.
        - Score new leads and export predictions."""
    )

    try:
        df = load_data()
    except Exception:
        st.error("UniversalBank.csv not found. Place it in the same folder as app.py.")
        return

    with st.expander("View Dataset Snapshot & Description"):
        st.dataframe(df.head())
        st.markdown(
            """**Key Fields**
            - Personal Loan (target), Age, Experience, Income, Family, CCAvg, Education, Mortgage
            - Securities Account, CD Account, Online, CreditCard"""
        )

    tab1, tab2, tab3 = st.tabs(
        ["📊 Customer Insights", "🤖 Model Performance", "📂 Upload & Predict"]
    )

    with tab1:
        customer_insights_tab(df)

    with tab2:
        st.subheader("Run & Compare Models")

        if st.button("Run All Models"):
            out = train_all_models(df)
            st.session_state["models"] = out["models"]
            st.session_state["metrics"] = out["metrics"]
            st.session_state["X_train"] = out["X_train"]
            st.session_state["X_test"] = out["X_test"]
            st.session_state["y_train"] = out["y_train"]
            st.session_state["y_test"] = out["y_test"]
            st.success("Models trained successfully.")

        if "metrics" in st.session_state:
            st.markdown("### Model Performance Summary")
            st.dataframe(st.session_state["metrics"], use_container_width=True)

            st.markdown("### ROC Curve (All Models)")
            plot_roc_curves(
                st.session_state["models"],
                st.session_state["X_test"],
                st.session_state["y_test"],
            )

            st.markdown("### Confusion Matrices")
            plot_confusion_matrices(
                st.session_state["models"],
                st.session_state["X_train"],
                st.session_state["X_test"],
                st.session_state["y_train"],
                st.session_state["y_test"],
            )

            st.markdown("### Feature Importance")
            feature_names = st.session_state["X_train"].columns
            plot_feature_importances(st.session_state["models"], feature_names)
        else:
            st.info("Click 'Run All Models' to train and evaluate models.")

    with tab3:
        models = st.session_state.get("models", None)
        upload_and_predict_tab(df, models)

if __name__ == "__main__":
    main()
