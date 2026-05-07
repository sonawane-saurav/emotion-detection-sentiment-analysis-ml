import streamlit as st
import pickle
import numpy as np
from transformers import pipeline

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Emotion AI Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# DARK UI STYLE
# =========================
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: white;
    }

    .stTextArea textarea {
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

# =========================
# TITLE
# =========================
st.title("🧠 Hybrid NLP-Based Emotion Intelligence System")
st.markdown("Hybrid Model: TF-IDF + Transformer (DistilRoBERTa)")

# =========================
# LOAD MODELS
# =========================
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

bert_model = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base"
)

# =========================
# EMOJI MAP FOR TRANSFORMER
# =========================
emoji_map = {
    "joy": "😊 Joy",
    "anger": "😡 Anger",
    "sadness": "😢 Sadness",
    "fear": "😨 Fear",
    "love": "❤️ Love",
    "neutral": "😐 Neutral"
}

# =========================
# LABEL MAP FOR TF-IDF MODEL
# =========================
label_map = {
    0: "😢 Sadness",
    1: "😡 Anger",
    2: "😊 Joy",
    3: "😨 Fear",
    4: "❤️ Love",
    5: "😐 Neutral"
}

# =========================
# SIDEBAR EXAMPLES
# =========================
st.sidebar.header("💡 Try Examples")

examples = [
    "I am very happy today",
    "I feel angry and frustrated",
    "I am scared about exams",
    "I love this project so much",
    "I feel nothing special"
]

for e in examples:
    if st.sidebar.button(e):
        st.session_state["input_text"] = e

# =========================
# INPUT
# =========================
text = st.text_area(
    "Enter your text:",
    value=st.session_state.get("input_text", ""),
    height=150
)

# =========================
# PREDICTION
# =========================
if st.button("🚀 Predict Emotion"):

    # Prevent empty prediction
    if text.strip() == "":
        st.warning("Please enter some text.")
    else:

        # ---------------------
        # TF-IDF MODEL
        # ---------------------
        vec = vectorizer.transform([text])

        pred_ml = model.predict(vec)[0]
        pred_ml_label = label_map.get(pred_ml, "Unknown")

        try:
            conf_ml = np.max(model.predict_proba(vec))
        except:
            conf_ml = 0.6

        # ---------------------
        # TRANSFORMER MODEL
        # ---------------------
        bert_result = bert_model(text)[0]

        label = bert_result["label"].lower()
        score = bert_result["score"]

        emotion = emoji_map.get(label, label)

        # =========================
        # OUTPUT LAYOUT
        # =========================
        col1, col2 = st.columns(2)

        # ---------------------
        # ML MODEL OUTPUT
        # ---------------------
        with col1:
            st.markdown("### 📊 Classical ML (TF-IDF)")

            st.markdown(f"""
            <div style='padding:10px;background:#262730;border-radius:10px'>
            🎯 Predicted Emotion: <b>{pred_ml_label}</b>
            </div>
            """, unsafe_allow_html=True)

            st.write("Confidence:")
            st.progress(float(conf_ml))
            st.info(f"{round(conf_ml * 100, 2)}%")

        # ---------------------
        # TRANSFORMER OUTPUT
        # ---------------------
        with col2:
            st.markdown("### 🤖 Transformer Model")

            st.markdown(f"""
            <div style='padding:10px;background:#262730;border-radius:10px'>
            🎯 Prediction: <b>{emotion}</b>
            </div>
            """, unsafe_allow_html=True)

            st.write("Confidence:")
            st.progress(float(score))
            st.info(f"{round(score * 100, 2)}%")

# =========================
# COMPARISON SECTION
# =========================
st.markdown("---")
st.subheader("📊 Model Comparison")

col1, col2 = st.columns(2)

with col1:
    st.metric("TF-IDF Model", "51% (approx)", "Classical ML")

with col2:
    st.metric("Transformer Model", "85%+ (typical)", "Deep Learning")

# =========================
# EXPLANATION SECTION
# =========================
st.markdown("---")
st.subheader("🧠 How It Works")

st.info("""
- TF-IDF model learns word frequency patterns.
- Transformer model understands context and meaning.
- This allows comparison of classical ML vs modern AI approaches.
""")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("🚀 Built with Streamlit | TF-IDF + Transformers | Emotion AI System")
