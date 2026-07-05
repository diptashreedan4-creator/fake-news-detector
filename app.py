import streamlit as st
import pickle
import re
import pandas as pd
from PIL import Image

st.set_page_config(page_title="Fake News Detector", layout="wide")

st.title("AI-Powered Fake News Detection")
st.write("A machine learning pipeline built using TF-IDF features and 4 classification models.")

@st.cache_resource
def load_models():
    with open("tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    models = {}
    for name in ["LogisticRegression", "RandomForest", "NeuralNetwork", "KNN"]:
        with open("model_" + name + ".pkl", "rb") as f:
            models[name] = pickle.load(f)
    return vectorizer, models

vectorizer, models = load_models()

stopwords = set("""
a an the and or but if while is are was were be been being
in on at to for of with as by from about into over after
this that these those it its i you he she they we
not no nor so than too very can will just should now
""".split())

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in stopwords and len(w) > 1]
    return " ".join(tokens)

tab1, tab2 = st.tabs(["Try the Demo", "Project Results"])

with tab1:
    st.header("Test an Article")
    model_choice = st.selectbox("Choose a model", list(models.keys()))
    user_text = st.text_area("Paste a news article or headline here:", height=200)

    if st.button("Check if Real or Fake"):
        if user_text.strip() == "":
            st.warning("Please enter some text first.")
        else:
            cleaned = clean_text(user_text)
            vec = vectorizer.transform([cleaned])
            model = models[model_choice]
            pred = model.predict(vec)[0]
            proba = model.predict_proba(vec)[0] if hasattr(model, "predict_proba") else None

            if pred == 1:
                st.error("Prediction: FAKE NEWS")
            else:
                st.success("Prediction: REAL NEWS")

            if proba is not None:
                st.write("Confidence -> Real: " + str(round(proba[0]*100, 2)) + "%, Fake: " + str(round(proba[1]*100, 2)) + "%")

with tab2:
    st.header("Model Performance Comparison")
    try:
        metrics_df = pd.read_csv("model_comparison_metrics.csv")
        st.dataframe(metrics_df)
    except:
        st.write("Metrics file not found.")

    col1, col2 = st.columns(2)
    with col1:
        try:
            st.image("model_accuracy_comparison.png", caption="Accuracy Comparison")
        except:
            pass
    with col2:
        try:
            st.image("model_all_metrics_comparison.png", caption="All Metrics Comparison")
        except:
            pass

    try:
        st.image("confusion_matrices_all_models.png", caption="Confusion Matrices")
    except:
        pass

    try:
        st.image("word_count_distribution.png", caption="Word Count Distribution")
    except:
        pass
