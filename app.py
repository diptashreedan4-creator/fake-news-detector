import streamlit as st
import pickle
import re
import pandas as pd

st.set_page_config(page_title="Fake News Detector", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Dashboard", "Project", "Contact"])

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

@st.cache_resource
def load_models():
    with open("tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    models = {}
    for name in ["LogisticRegression", "RandomForest", "NeuralNetwork", "KNN"]:
        with open("model_" + name + ".pkl", "rb") as f:
            models[name] = pickle.load(f)
    return vectorizer, models

if page == "Home":
    st.title("AI-Powered Fake News Detection")
    st.subheader("A Machine Learning Project using TF-IDF and Text Classification")
    st.write("")
    st.write("This project builds a complete machine learning pipeline from scratch to classify news articles as Real or Fake.")
    st.write("")
    st.markdown("### What this project covers:")
    st.markdown("- Data cleaning and manual text preprocessing")
    st.markdown("- TF-IDF feature engineering and exploratory data analysis")
    st.markdown("- Four classification models: KNN, Logistic Regression, Random Forest, and Neural Network")
    st.markdown("- Full evaluation with accuracy, precision, recall, F1-score, and confusion matrices")
    st.write("")
    st.info("Use the sidebar on the left to explore the Dashboard, try the live Project demo, or find Contact details.")

elif page == "Dashboard":
    st.title("Project Dashboard")
    st.write("Model performance comparison across all four algorithms.")

    try:
        metrics_df = pd.read_csv("model_comparison_metrics.csv")
        st.dataframe(metrics_df, use_container_width=True)
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

elif page == "Project":
    st.title("Try the Fake News Detector")
    vectorizer, models = load_models()

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

elif page == "Contact":
    st.title("Contact")
    st.write("Feel free to reach out with any questions, feedback, or collaboration ideas.")
    st.write("")
    st.markdown("**Email:** diptashreedan4@gmail.com")

