import os

import pandas as pd
import requests
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


st.set_page_config(
    page_title="Credit Card Fraud Detection",
    layout="wide",
)

st.title("Credit Card Fraud Detection System")

st.write(
    "Upload a compatible transaction CSV to calculate model "
    "fraud-risk scores and predicted classes."
)

default_api_url = os.getenv(
    "API_BASE_URL",
    "http://127.0.0.1:8000",
)

api_url = st.text_input(
    "FastAPI base URL",
    value=default_api_url,
).rstrip("/")

uploaded_file = st.file_uploader(
    "Upload transaction CSV",
    type=["csv"],
)

if uploaded_file is not None:
    try:
        input_df = pd.read_csv(uploaded_file)

    except Exception:
        st.error("Could not read the uploaded CSV.")
        st.stop()

    if input_df.empty:
        st.error("The CSV contains no transactions.")
        st.stop()

    # If this is the public labelled dataset, keep the true
    # labels separate from the model input.
    actual_labels = None

    if "Class" in input_df.columns:
        actual_labels = input_df["Class"].copy()
        input_df = input_df.drop(columns=["Class"])

        st.info(
            "The dataset's Class column will not be sent to the model. "
            "It can be displayed separately for demonstration."
        )

    st.subheader("Preview of input features")
    st.dataframe(input_df.head())

    if st.button("Run fraud predictions", type="primary"):
        csv_bytes = input_df.to_csv(index=False).encode("utf-8")

        try:
            response = requests.post(
                f"{api_url}/predict-file",
                files={
                    "file": (
                        "transactions.csv",
                        csv_bytes,
                        "text/csv",
                    )
                },
                timeout=120,
            )

            response.raise_for_status()
            payload = response.json()

            result_df = pd.DataFrame(payload["results"])

            def get_risk_level(score):
                if score < 0.30:
                    return "Low"
                elif score < 0.70:
                    return "Medium"
                return "High"


            result_df["risk_level"] = result_df["risk_score"].apply(
                get_risk_level
            )

            if actual_labels is not None:
                result_df["dataset_label"] = (
                    actual_labels.reset_index(drop=True)
                    .map({
                        0: "legitimate",
                        1: "fraud",
                    })
                    .to_numpy()
                )

            if actual_labels is not None:
                actual = actual_labels.reset_index(drop=True).astype(int)

                predicted = (
                    result_df["prediction"]
                    .map({
                        "legitimate": 0,
                        "fraud": 1,
                    })
                    .astype(int)
                )

                accuracy = accuracy_score(actual, predicted)
                precision = precision_score(
                    actual,
                    predicted,
                    zero_division=0,
                )
                recall = recall_score(
                    actual,
                    predicted,
                    zero_division=0,
                )
                f1 = f1_score(
                    actual,
                    predicted,
                    zero_division=0,
                )

                st.subheader("Model evaluation")

                col1, col2, col3, col4 = st.columns(4)

                col1.metric("Accuracy", f"{accuracy:.2%}")
                col2.metric("Precision", f"{precision:.2%}")
                col3.metric("Recall", f"{recall:.2%}")
                col4.metric("F1 Score", f"{f1:.2%}")

                cm = confusion_matrix(actual, predicted)

                st.write("Confusion Matrix")

                cm_df = pd.DataFrame(
                    cm,
                    index=["Actual Legitimate", "Actual Fraud"],
                    columns=["Predicted Legitimate", "Predicted Fraud"],
                )

                st.dataframe(cm_df)

            st.subheader("Prediction results")
            st.subheader("Prediction distribution")

            prediction_counts = result_df["prediction"].value_counts()

            st.bar_chart(prediction_counts)

            first, second, third, fourth = st.columns(4)

            first.metric(
                "Rows processed",
                payload["rows_processed"],
            )

            second.metric(
                "Flagged as fraud",
                payload["fraud_flagged"],
            )

            third.metric(
                "Decision threshold",
                f"{payload['threshold']:.2f}",
            )

            fourth.metric(
                "Fraud rate",
                f"{payload['fraud_flagged'] / payload['rows_processed'] * 100:.2f}%"
            )

            display_df = result_df.sort_values(
                "risk_score",
                ascending=False,
            )

            st.dataframe(
                display_df,
                use_container_width=True,
            )

            csv_output = result_df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="Download prediction results",
                data=csv_output,
                file_name="fraud_predictions.csv",
                mime="text/csv",
            )

        except requests.RequestException as exc:
            st.error(
                "Could not get predictions from the backend. "
                f"Check the API URL and server. Details: {exc}"
            )