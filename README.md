# Phân tích cảm xúc – Sentiment Analysis với Naive Bayes

- Dự án này phân tích cảm xúc (positive/negative) từ các bình luận trên trang thương mại điện tử sử dụng mô hình Naive Bayes.  
- Có hai phiên bản mô hình: nhỏ (1000 reviews) và lớn (23000 reviews).  

---

## Cài đặt

1. Clone repository  
```bash
git clone https://github.com/FairySS1210/Presentation_Code_Nhom8_SentimentAnalysis_NaiveBayes_VoThuTien.git
cd phan_tich_cam_xuc
```

2. Tạo môi trường ảo  
```bash
python -m venv venv
venv\Scripts\activate
```

3. Cài đặt các thư viện cần thiết  
```bash
pip install -r requirements.txt
```

---

## Cách chạy

1. Chạy ứng dụng Streamlit (run app.py)  
```bash
streamlit run app.py
```

2. Dự đoán trực tiếp với file Python  
```bash
python predict_demo_small.py   # model nhỏ
python predict_demo_large.py   # model lớn
```

3. Huấn luyện lại mô hình  
```bash
python train_model_small.py
python train_model_large.py
```

---

## Dữ liệu

Dữ liệu bình luận được lưu trong thư mục `data/`.  
Các file `.csv` đã được clean để phục vụ huấn luyện mô hình.  
