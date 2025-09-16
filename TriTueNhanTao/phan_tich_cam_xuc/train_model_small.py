import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import joblib
import os

# Đường dẫn
CSV_PATH = "data/reviews_1000_clean.csv"
MODEL_DIR = "model_small"

# Đảm bảo thư mục model tồn tại
os.makedirs(MODEL_DIR, exist_ok=True)

# Đọc dữ liệu
df = pd.read_csv(CSV_PATH)

# Tách dữ liệu
X = df["Review Text"]
y = df["Label"]

# Chia train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Vector hóa TF-IDF
vectorizer = TfidfVectorizer(stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Naive Bayes
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Đánh giá
y_pred = model.predict(X_test_vec)
print("Đánh giá trên dataset nhỏ (1000 dòng):")
print(classification_report(y_test, y_pred))

# Lưu model và vectorizer
joblib.dump(model, os.path.join(MODEL_DIR, "nb_model.joblib"))
joblib.dump(vectorizer, os.path.join(MODEL_DIR, "vectorizer.joblib"))

print(f"✅ Model nhỏ đã lưu tại {MODEL_DIR}")
