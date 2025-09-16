import os
import re
import joblib
import streamlit as st

# ---------------- Hàm tiền xử lý ----------------
def clean_text(s):
    s = s.lower()
    s = re.sub(r"[^a-z\s]", " ", s)  # giữ chữ thường a-z và khoảng trắng
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ---------------- Hàm dự đoán ----------------
def predict_sentiment(text, model, vectorizer):
    text_clean = clean_text(text)
    vec = vectorizer.transform([text_clean])
    pred = model.predict(vec)[0]
    return pred

# ---------------- Giao diện ----------------
st.set_page_config(page_title="Sentiment Analysis", page_icon="🎯", layout="wide")

# Người dùng chọn model
option = st.radio("Sentiment analysis of product reviews", ("Dataset nhỏ (1k)", "Dataset lớn (23k)"))

st.markdown("<hr>", unsafe_allow_html=True)

if option == "Dataset nhỏ (1k)":
    MODEL_DIR = "model_small"
    IMAGE_FILE = "phone.png"
    CAPTION = "Samsung Galaxy Z Flip4 5G 128GB"
    STATE_KEY = "reviews_small"
else:
    MODEL_DIR = "model_large"
    IMAGE_FILE = "phone2.png"
    CAPTION = "Samsung Galaxy Y Flip4 5G 128GB"
    STATE_KEY = "reviews_large"

# Khởi tạo state để lưu bình luận (đảm bảo luôn chạy trước khi truy cập)
if STATE_KEY not in st.session_state:
    st.session_state[STATE_KEY] = []

# Load model & vectorizer
MODEL_FILE = os.path.join(MODEL_DIR, "nb_model.joblib")
VECT_FILE = os.path.join(MODEL_DIR, "vectorizer.joblib")

model = joblib.load(MODEL_FILE)
vectorizer = joblib.load(VECT_FILE)

# Create two columns
col1, col2 = st.columns([1, 1])

# Left column: Image and Input Form
with col1:
    # Hiển thị ảnh minh họa
    img_path = os.path.join(os.path.dirname(__file__), IMAGE_FILE)
    if os.path.exists(img_path):
        st.image(img_path, caption=CAPTION, width=230)
    else:
        st.warning(f"❌ {IMAGE_FILE}")

    # Ô nhập bình luận
    user_input = st.text_area("Enter comment", "", height=100)
    if st.button("Send"):
        if user_input.strip() == "":
            st.warning("⚠️")
        else:
            sentiment = predict_sentiment(user_input, model, vectorizer)
            st.session_state[STATE_KEY].append((user_input, sentiment))

# Right column: Customer Reviews
with col2:
    with st.container(height=400):
        if st.session_state[STATE_KEY]:
            for comment, senti in reversed(st.session_state[STATE_KEY]):
                if senti.lower() == "positive":
                    st.success(f"**{comment}** → Tích cực 👍")
                elif senti.lower() == "negative":
                    st.error(f"**{comment}** → Tiêu cực 👎")
                else:
                    st.info(f"**{comment}** → Trung lập 😐")
        else:
            st.write("...")