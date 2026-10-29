
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# Use the folder containing this app.py file
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "machine_failure_model.pkl"

FEATURE_COLUMNS = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

st.set_page_config(
    page_title="Machine Failure Prediction System",
    page_icon="⚙️",
    layout="wide",
)

st.title("⚙️ Machine Failure Prediction System")
st.caption("Machine health analysis and failure prediction")

# Check whether the trained model exists
if not MODEL_PATH.exists():
    st.error("Trained model file not found!")
    st.write(
        "Expected location:",
        str(MODEL_PATH),
    )
    st.warning(
        "First run train_model.py successfully. "
        "It must save machine_failure_model.pkl "
        "in the same folder as app.py."
    )
    st.code(
        "python train_model.py\n"
        "python -m streamlit run app.py",
        language="bash",
    )
    st.stop()

# Load the trained model safely
try:
    model = joblib.load(MODEL_PATH)
except Exception as error:
    st.error("The saved model could not be loaded.")
    st.code(str(error))
    st.stop()

page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Predict Failure", "Dataset Analysis"],
)

if page == "Dashboard":
    st.header("Machine Monitoring Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric("Model Status", "Loaded")
    col2.metric("Input Features", len(FEATURE_COLUMNS))
    col3.metric("Prediction System", "Ready")

    st.info(
        "Open Predict Failure to enter machine sensor "
        "readings, or Dataset Analysis to upload a CSV."
    )

elif page == "Predict Failure":
    st.header("Predict Machine Failure")

    with st.form("failure_prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            machine_type = st.selectbox(
                "Machine Type",
                ["L", "M", "H"],
            )

            air_temp = st.number_input(
                "Air Temperature (K)",
                min_value=250.0,
                max_value=350.0,
                value=298.1,
            )

            process_temp = st.number_input(
                "Process Temperature (K)",
                min_value=250.0,
                max_value=400.0,
                value=308.6,
            )

        with col2:
            speed = st.number_input(
                "Rotational Speed (rpm)",
                min_value=1,
                max_value=10000,
                value=1500,
            )

            torque = st.number_input(
                "Torque (Nm)",
                min_value=0.0,
                max_value=200.0,
                value=40.0,
            )

            tool_wear = st.number_input(
                "Tool Wear (minutes)",
                min_value=0,
                max_value=300,
                value=10,
            )

        submitted = st.form_submit_button(
            "Predict Failure",
            type="primary",
        )

    if submitted:
        input_data = pd.DataFrame(
            [[
                machine_type,
                air_temp,
                process_temp,
                speed,
                torque,
                tool_wear,
            ]],
            columns=FEATURE_COLUMNS,
        )

        try:
            prediction = int(model.predict(input_data)[0])

            st.subheader("Prediction Result")

            if prediction == 1:
                st.error("Potential Machine Failure Detected")
                st.write(
                    "Recommendation: arrange an inspection "
                    "and check the machine's operating condition."
                )
            else:
                st.success("No Failure Predicted")

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_data)[0]
                classes = list(model.classes_)

                if 1 in classes:
                    risk = float(
                        probabilities[classes.index(1)] * 100
                    )

                    st.metric(
                        "Model-Estimated Failure Probability",
                        f"{risk:.2f}%",
                    )

                    st.progress(risk / 100)

            with st.expander("View Input Sensor Readings"):
                st.dataframe(
                    input_data,
                    use_container_width=True,
                )

            st.caption(
                "Predictions are estimates from historical data "
                "and do not guarantee the actual machine condition."
            )

        except Exception as error:
            st.error("Prediction failed.")
            st.code(str(error))
            st.info(
                "Verify that the trained model uses the same "
                "six feature columns and preprocessing."
            )

elif page == "Dataset Analysis":
    st.header("Dataset Analysis")

    uploaded_file = st.file_uploader(
        "Upload machine dataset (CSV)",
        type=["csv"],
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            df.columns = df.columns.str.strip()

            col1, col2 = st.columns(2)
            col1.metric("Total Records", len(df))
            col2.metric("Total Columns", len(df.columns))

            st.subheader("Dataset Preview")
            st.dataframe(
                df.head(20),
                use_container_width=True,
            )

            st.subheader("Missing Values")
            st.dataframe(
                df.isnull().sum().rename("Missing Values")
            )

            st.subheader("Statistical Summary")
            st.dataframe(
                df.describe(),
                use_container_width=True,
            )

            numeric_columns = df.select_dtypes(
                include="number"
            ).columns.tolist()

            if numeric_columns:
                selected_column = st.selectbox(
                    "Choose a numeric feature to visualize",
                    numeric_columns,
                )

                st.subheader(f"Distribution: {selected_column}")
                st.bar_chart(
                    df[selected_column].value_counts().sort_index()
                )

            if st.button("Predict Uploaded Records"):
                missing = [
                    col for col in FEATURE_COLUMNS
                    if col not in df.columns
                ]

                if missing:
                    st.error(
                        "Missing required columns: "
                        + ", ".join(missing)
                    )
                else:
                    predictions = model.predict(
                        df[FEATURE_COLUMNS]
                    )

                    results = df.copy()
                    results["Predicted Failure"] = predictions

                    if hasattr(model, "predict_proba"):
                        probabilities = model.predict_proba(
                            df[FEATURE_COLUMNS]
                        )
                        classes = list(model.classes_)

                        if 1 in classes:
                            results[
                                "Failure Probability (%)"
                            ] = (
                                probabilities[:, classes.index(1)]
                                * 100
                            ).round(2)

                    st.subheader("Prediction Results")
                    st.dataframe(
                        results,
                        use_container_width=True,
                    )

                    csv_bytes = results.to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(
                        "Download Prediction Results",
                        data=csv_bytes,
                        file_name="machine_predictions.csv",
                        mime="text/csv",
                    )

        except Exception as error:
            st.error("Could not process the uploaded CSV.")
            st.code(str(error))
