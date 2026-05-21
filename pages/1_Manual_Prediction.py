import os
import streamlit as st
import tensorflow as tf
import numpy as np
import random
from transformers import AutoTokenizer
from transformers import TFAutoModel

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from utils import load_model_and_tokenizer

tokenizer, model = load_model_and_tokenizer()

st.title("⚖️ Manual Legal Prediction")

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
label_map = {0: "Neutral ⚖️", 1: "Entailment ✅", 2: "Contradiction ❌"}

with st.expander("ℹ️ How the model works"):
    st.write(
        "This AI model analyzes two legal statements and predicts whether they are:\n\n"
        "✅ Entailment — both statements support each other \n\n"
        "⚖️ Neutral — statements are unrelated \n\n"
        "❌ Contradiction — statements conflict with each other"
    )
samples = [
    {
        "premise": "The company signed the agreement in 2022.",
        "hypothesis": "The agreement contains clauses related to taxation.",
    },
    {
        "premise": "The judge approved the merger between the two companies.",
        "hypothesis": "The merger received judicial approval.",
    },
    {
        "premise": "The agreement was signed by both parties.",
        "hypothesis": "The contract was never approved.",
    },
]

if "premise" not in st.session_state:
    st.session_state.premise = ""

if "hypothesis" not in st.session_state:
    st.session_state.hypothesis = ""

if st.button("🧪 Try Sample"):
    sample = random.choice(samples)
    st.session_state.premise = sample["premise"]
    st.session_state.hypothesis = sample["hypothesis"]

sentence1 = st.text_area(
    "📄 Premise (Original Legal Statement)",
    key="premise",
    height=120,
    placeholder="Example: The agreement was signed by both parties on 12 March 2022.",
    help="Enter the original legal statement, clause, or document sentence.",
)

sentence2 = st.text_area(
    "🔍 Hypothesis (Statement to Compare)",
    key="hypothesis",
    height=120,
    placeholder="Example: The contract was never approved by the parties.",
    help="Enter the second statement you want to compare against the premise.",
)
col1, col2 = st.columns(2)
with col1:
    if st.button("🔮 Predict Contradiction"):

        if sentence1.strip() == "" or sentence2.strip() == "":
            st.warning("⚠️ Please enter both legal statements.")
        else:

            with st.spinner("Analyzing legal statements..."):

                inputs = tokenizer(
                    sentence1,
                    sentence2,
                    padding="max_length",
                    truncation=True,
                    max_length=128,
                    return_tensors="tf",
                )

                prediction = model.predict(
                    {
                        "input_ids": inputs["input_ids"],
                        "attention_mask": inputs["attention_mask"],
                    },
                    verbose=0,
                )

                predicted_class = np.argmax(prediction, axis=1)[0]

                st.success(f"Prediction: {label_map[predicted_class]}")
with col2:
    if st.button("Go to Bulk Prediction"):
        st.write("Redirecting to Bulk Prediction page...")
        st.switch_page("pages/2_Bulk_Scanner.py")
