import joblib
import os

# Đường dẫn model
MODEL_DIR = "model_large"
MODEL_FILE = os.path.join(MODEL_DIR, "nb_model.joblib")
VECT_FILE = os.path.join(MODEL_DIR, "vectorizer.joblib")

# Load model và vectorizer
model = joblib.load(MODEL_FILE)
vectorizer = joblib.load(VECT_FILE)

# Hàm dự đoán
def predict_sentiment(text):
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)[0]
    return prediction

# Ví dụ
examples = [
    "This dress looks amazing",
    "Ugly, not worth for money"
]

for ex in examples:
    print(f"👉 {ex} => {predict_sentiment(ex)}")
