# app.py

import streamlit as st
from model_backend import SentimentModel
import os

MODEL_NAME_OR_PATH = "ai4bharat/indic-bert"
WEIGHTS_PATH = "indic_bert_sentiment_model.pth" #a local path or sathyavgc/indic-bert-sentiment-malayalam from HF
LABEL_MAP = {
    "negative": 0,
    "positive": 1
}
MAX_LENGTH = 128

os.environ['HF_TOKEN'] = "HF_TOKEN"

@st.cache_resource
def load_model():
    return SentimentModel(
        model_name_or_path=MODEL_NAME_OR_PATH,
        label_map=LABEL_MAP,
        max_length=MAX_LENGTH,
        weights_path=WEIGHTS_PATH
    )

model = load_model()

st.set_page_config(page_title="മലയാളം സിനിമ റിവ്യൂ Sentiment", layout="centered")
st.title("മലയാളം സിനിമ റിവ്യൂ – Sentiment Analysis")
st.write("ദയവായി മലയാളത്തില്‍ മാത്രം റിവ്യൂ എഴുതി വിശദമായി അടയാളപ്പെടുത്തൂ:")

review_text = st.text_area("റിവ്യൂ:", height=150, placeholder="ഇവിടെ നിങ്ങളുടെ റിവ്യൂ എഴുതൂ…")

if st.button("അനാലൈസ് ചെയ്യുക"):
    if not review_text.strip():
        st.warning("ദയവായി റിവ്യൂ എഴുതുക.")
    else:
        result = model.predict(review_text.strip())
        if result == "ERROR_NOT_MALAYALAM":
            st.warning("ദയവായി മലയാളത്തിലാകുന്ന റിവ്യൂ മാത്രം നല്‍കുക.")
        else:
            if result == "positive":
                st.success(f"പ്രകടനം: **{result}**")
            elif result == "negative":
                st.error(f"പ്രകടനം: **{result}**")
            else:
                st.info(f"പ്രകടനം: **{result}**")


