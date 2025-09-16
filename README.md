Phân tích cảm xúc - Sentiment Analysis với Naive Bayes
Dự án này phân tích cảm xúc (positive/negative) từ các bình luận trên trang thương mại điện tử sử dụng mô hình Naive Bayes.
Có hai phiên bản mô hình: nhỏ (1000 reviews) và lớn (23000 reviews).
Cài đặt
Clone repository
git clone https://github.com/FairySS1210/Presentation_Code_Nhom8_SentimentAnalysis_NaiveBayes_VoThuTien.git

cd phan_tich_cam_xuc

Tạo môi trường ảo
python -m venv venv

venv\Scripts\activate

Cài đặt các thư viện cần thiết
pip install -r requirements.txt

Cách chạy
Chạy ứng dụng Streamlit (run app.py)
streamlit run app.py

Dự đoán trực tiếp với file Python
python predict_demo_small.py - model nhỏ

python predict_demo_large.py - model lớn

Huấn luyện lại mô hình
python train_model_small.py

python train_model_large.py

Dữ liệu
Dữ liệu bình luận được lưu trong thư mục data/. Các file .csv đã được clean để phục vụ huấn luyện mô hình.
