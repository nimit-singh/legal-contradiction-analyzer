import streamlit as st
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from transformers import AutoTokenizer
from transformers import TFAutoModel


@st.cache_resource
def load_model_and_tokenizer():
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained("legal_tokenizer")
    bert_model = TFAutoModel.from_pretrained(MODEL_NAME)
    return tokenizer, bert_model
