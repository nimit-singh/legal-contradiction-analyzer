import streamlit as st
import tensorflow as tf

tf.get_logger().setLevel("ERROR")
import pandas as pd
import numpy as np
import random
from transformers import AutoTokenizer
from transformers import TFAutoModel
import json
from io import BytesIO
import openpyxl
import os
from utils import load_model_and_tokenizer

bytes = BytesIO()
tokenizer, model = load_model_and_tokenizer()
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


st.set_page_config(layout="centered")
st.title("Legal  Bulk Scanner")
with st.expander("ℹ️ How Bulk Scanner Works"):
    st.markdown("""
    ### 📌 Instructions

    Upload a CSV, Excel, or JSON file containing:

    - `Premise`
    - `Hypothesis`

    columns.

    The AI model will analyze each row and classify it as:

    - ✅ Entailment
    - ⚖️ Neutral
    - ❌ Contradiction

    ### ✅ Supported Formats
    - CSV
    - Excel (.xlsx)
    - JSON

    ### 📄 Example Structure

    | Premise | Hypothesis |
    |----------|------------|
    | Agreement was signed | Contract was approved |

    """)


st.markdown("---")
st.header("🔎 Bulk Premium Scanner")
input_ids_layer = tf.keras.layers.Input(shape=(128,), dtype=tf.int32, name="input_ids")

attention_mask_layer = tf.keras.layers.Input(
    shape=(128,), dtype=tf.int32, name="attention_mask"
)

outputs = model(input_ids_layer, attention_mask=attention_mask_layer)

x = outputs.last_hidden_state[:, 0, :]

x = tf.keras.layers.Dropout(0.2)(x)

x = tf.keras.layers.Dense(128, activation="relu")(x)

x = tf.keras.layers.Dropout(0.3)(x)

output = tf.keras.layers.Dense(3, activation="softmax")(x)

model = tf.keras.Model(inputs=[input_ids_layer, attention_mask_layer], outputs=output)

model.load_weights("model_weights.h5")

label_map = {0: "Neutral", 1: "Entailment", 2: "Contradiction"}
col1, col2, col3 = st.columns(3)

sample_csv = pd.DataFrame(
    [
        {
            "Premise": "The company signed the agreement in 2022.",
            "Hypothesis": "The agreement contains clauses related to taxation.",
        },
        {
            "Premise": "The judge approved the merger between the two companies.",
            "Hypothesis": "The merger received judicial approval.",
        },
        {
            "Premise": "The agreement was signed by both parties.",
            "Hypothesis": "The contract was never approved.",
        },
    ]
)
sample_excel = pd.DataFrame(
    [
        {
            "Premise": "The company signed the agreement in 2022.",
            "Hypothesis": "The agreement contains clauses related to taxation.",
        },
        {
            "Premise": "The judge approved the merger between the two companies.",
            "Hypothesis": "The merger received judicial approval.",
        },
        {
            "Premise": "The agreement was signed by both parties.",
            "Hypothesis": "The contract was never approved.",
        },
    ]
)
sample_json = pd.DataFrame(
    [
        {
            "Premise": "The company signed the agreement in 2022.",
            "Hypothesis": "The agreement contains clauses related to taxation.",
        },
        {
            "Premise": "The judge approved the merger between the two companies.",
            "Hypothesis": "The merger received judicial approval.",
        },
        {
            "Premise": "The agreement was signed by both parties.",
            "Hypothesis": "The contract was never approved.",
        },
    ]
)

csv_data = sample_csv.to_csv(index=False)
with col1:
    st.download_button(
        label="Sample CSV",
        data=csv_data,
        file_name="sample.csv",
        mime="text/csv",
        width=200,
    )
sample_excel = sample_excel.to_excel(bytes, index=False)
with col2:
    st.download_button(
        label="Sample Excel",
        data=bytes.getvalue(),
        file_name="sample.xlsx",
        width=200,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
json_data = sample_json.to_json(orient="records", indent=4)
with col3:
    st.download_button(
        label="Sample JSON",
        data=json_data,
        file_name="sample.json",
        mime="application/json",
        width=200,
    )

st.markdown("---")

st.header("2. Upload Your File")
with st.expander("📂 Upload Instructions"):

    st.markdown("""
    ### Supported File Formats
    - CSV
    - Excel (.xlsx)
    - JSON

    ### Required Columns
    - Premise
    - Hypothesis

    ### Example
    | Premise | Hypothesis |
    |----------|------------|
    | Agreement signed | Contract approved |
    """)
st.write(
    "Upload a CSV, Excel, or JSON file with 'Premise' and 'Hypothesis' columns to get predictions in bulk."
)
uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "json"])

label_map = {0: "Neutral ⚖️", 1: "Entailment ✅", 2: "Contradiction ❌"}

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file, encoding="utf-8")
        st.write("Preview of Uploaded Data:")
        st.dataframe(df.head())
    elif uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
        st.write("Preview of Uploaded Data:")
        st.dataframe(df.head())
    elif uploaded_file.name.endswith(".json"):
        data = json.load(uploaded_file)
        df = pd.DataFrame(data)
        st.write("Preview of Uploaded Data:")
        st.dataframe(df.head())
    else:
        st.error("Unsupported file type. Please upload a CSV, Excel, or JSON file.")
        st.stop()

    if "Premise" not in df.columns or "Hypothesis" not in df.columns:
        st.error("File must contain 'Premise' and 'Hypothesis' columns.")
        st.stop()

    if st.button("Predict in Bulk"):
        inputs = tokenizer(
            df["Premise"].tolist(),
            df["Hypothesis"].tolist(),
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="tf",
        )

        prediction = model.predict(
            {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
            }
        )

        predicted_class = np.argmax(prediction, axis=1)

        df["Prediction"] = pd.Series(predicted_class).map(label_map)

        st.success("Bulk prediction completed!")

        st.write("Predictions:")

        st.dataframe(df[["Premise", "Hypothesis", "Prediction"]])

        output_csv = df[["Premise", "Hypothesis", "Prediction"]].to_csv(index=False)

        st.header("3. Download Your Predictions")

        st.success("Click the button below to download predictions as a CSV file.")

        st.download_button(
            label="Download Predictions as CSV",
            data=output_csv,
            file_name="predictions.csv",
            mime="text/csv",
        )
