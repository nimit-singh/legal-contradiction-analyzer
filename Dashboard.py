import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Legal NLI Dashboard", page_icon="⚖️")

st.title("⚖️ Legal Document Contradiction System")

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Model Accuracy", "89%")

with col2:
    st.metric("Classes", "3")

with col3:
    st.metric("Model Type", "Transformer")

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Model Size", "Small")

with col2:
    st.metric("Speed", "Fast")

with col3:
    st.metric("Architecture", "MiniLM")
st.markdown("---")

st.header("Project Overview")

st.write("""
This project detects relationships between legal statements:

- Entailment
- Neutral
- Contradiction

Built using:
- TensorFlow
- Transformer Architecture
- HuggingFace
- Streamlit
""")

st.markdown("---")
st.subheader("📊 Dataset Statistics")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("❌ Contradiction", "16,456")

with col2:
    st.metric("✅ Entailment", "16,456")

with col3:
    st.metric("⚖️ Neutral", "16,456")

sizes = [16456, 16456, 16456]

labels = ["Contradiction", "Entailment", "Neutral"]

colors = ["#ef4444", "#22c55e", "#3b82f6"]

fig, ax = plt.subplots(figsize=(7, 7))

# Remove white background
fig.patch.set_facecolor("#0e1117")
ax.set_facecolor("#0e1117")

wedges, texts, autotexts = ax.pie(
    sizes,
    labels=labels,
    colors=colors,
    autopct="%1.1f%%",
    startangle=90,
    textprops={"color": "white"},
)

# Legend styling
legend = ax.legend(
    wedges, labels, title="Classes", loc="center left", bbox_to_anchor=(1, 0.5)
)

plt.setp(legend.get_texts())
plt.setp(legend.get_title())

ax.axis("equal")

st.pyplot(fig)

st.header("Available Features")

st.write("✅ Manual Prediction")
st.write("✅ Bulk Prediction")
st.write("✅ CSV Upload")
st.write("✅ Download Predictions")
st.write("✅ Model Information")

st.markdown("---")

st.caption(
    "AI-Powered Legal Contradiction Detection using MiniLM + TensorFlow + Streamlit"
)

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    if st.button("Go to Manual Prediction"):
        st.switch_page("pages/1_Manual_Prediction.py")

with col2:
    if st.button("Go to Bulk Prediction"):
        st.switch_page("pages/2_Bulk_Scanner.py")
