import streamlit as st
import pickle
import os
import re
import string
import json
import base64
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer

# ------------------------------------
# STREAMLIT CONFIG
# ------------------------------------
st.set_page_config(page_title="Movie Sentiment AI", page_icon="🍿", layout="centered")

# ------------------------------------
# MODEL LOADING
# ------------------------------------
@st.cache_resource
def load_assets():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    deploy_dir = os.path.join(base_dir, "deployment")
    
    model_path = os.path.join(deploy_dir, "sentiment_model.pkl")
    vec_path = os.path.join(deploy_dir, "vectorizer.pkl")
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(vec_path, "rb") as f:
        vectorizer = pickle.load(f)
        
    return model, vectorizer

# ------------------------------------
# TEXT CLEANING
# ------------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ------------------------------------
# BACKGROUND SETUP
# ------------------------------------
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(image_file):
    bin_str = get_base64_of_bin_file(image_file)
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: linear-gradient(rgba(0, 0, 0, 0.6), rgba(0, 0, 0, 0.8)),
                          url("data:image/jpeg;base64,{bin_str}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    h1, h2, h3, p, label, .stMarkdown, .stCaption {{
        color: white !important;
        text-shadow: 2px 2px 4px #000000;
    }}
    .stTextArea, .stButton {{
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 10px;
    }}
    .stButton>button {{
        color: white;
        background-color: #e50914;
        border: none;
        font-size: 18px;
        font-weight: bold;
    }}
    .stButton>button:hover {{
        background-color: #b20710;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)


# ------------------------------------
# MAIN APP
# ------------------------------------
def main():

    # Set background poster
    base_dir = os.path.dirname(os.path.abspath(__file__))
    poster_path = os.path.join(base_dir, 'wallpaper.jpg')

    try:
        set_background(poster_path)
    except FileNotFoundError:
        st.warning("⚠️ Could not find background poster.")

    # Navigation dropdown
    page = st.selectbox("Navigation", ["Sentiment Analyzer", "Model Performance"])

    # -------------------------
    # PAGE 1: SENTIMENT ANALYZER
    # -------------------------
    if page == "Sentiment Analyzer":

        st.title("🍿 Movie Review Analyzer")
        st.markdown("### Will you **Love** or **Hate** this movie? Let AI decide.")

        try:
            model, vectorizer = load_assets()
        except:
            st.error("⚠️ Could not load model or vectorizer.")
            return

        user_review = st.text_area(
            "📝 Write your review here:", 
            height=150, 
            placeholder="e.g. The visuals were stunning..."
        )

        if st.button("🎬 Analyze Sentiment"):
            if user_review.strip():
                cleaned_text = clean_text(user_review)
                vec_input = vectorizer.transform([cleaned_text])

                prediction = model.predict(vec_input)[0]
                probs = model.predict_proba(vec_input)[0]

                st.markdown("---")
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    if prediction == 1:
                        st.success("## 🎉 POSITIVE")
                        st.snow()
                    else:
                        st.error("## 🍅 NEGATIVE")
                
                with col2:
                    st.write("### Confidence Score")
                    confidence = probs[prediction]
                    st.progress(confidence)
                    st.caption(f"The AI is **{confidence:.2%}** sure about this result.")
            else:
                st.warning("Please enter a review first!")

    # -------------------------
    # PAGE 2: PERFORMANCE METRICS
    # -------------------------
    elif page == "Model Performance":

        st.title("📊 Model Performance Metrics")

        deploy_dir = os.path.join(base_dir, "deployment")
        models_dir = os.path.join(base_dir, "models")

        # Load metadata.json
        meta_path = os.path.join(deploy_dir, "metadata.json")
        report_path = os.path.join(deploy_dir, "report.json")
        cm_path = os.path.join(models_dir, "confusion_matrix.png")

        if not os.path.exists(meta_path) or not os.path.exists(report_path):
            st.error("⚠️ Missing metadata.json or report.json. Retrain the model with saving enabled.")
            return

        with open(meta_path, "r") as f:
            meta = json.load(f)

        with open(report_path, "r") as f:
            report = json.load(f)

        # Extract metrics
        accuracy = meta.get("accuracy", None)
        precision = report['weighted avg']['precision']
        recall = report['weighted avg']['recall']
        f1 = report['weighted avg']['f1-score']

        # Display metric cards
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{accuracy:.2%}")
        col2.metric("Precision", f"{precision:.2%}")
        col3.metric("Recall", f"{recall:.2%}")

        col4, = st.columns(1)
        col4.metric("F1 Score", f"{f1:.2%}")

        st.markdown("---")

        # Display Confusion Matrix
        st.subheader("📏 Confusion Matrix")
        if os.path.exists(cm_path):
            st.image(cm_path, caption="Confusion Matrix")
        else:
            st.warning("⚠️ Confusion matrix image not found.")


if __name__ == "__main__":
    main()
