import pandas as pd

# Đọc file gốc
df = pd.read_csv("data/Womens Clothing E-Commerce Reviews.csv")

# Giữ lại 2 cột cần thiết
df = df[['Review Text', 'Rating']].dropna()

# Hàm ánh xạ Rating → Label
def to_sentiment(rating):
    if rating in [1, 2]:
        return "Negative"
    elif rating == 3:
        return "Neutral"
    else:
        return "Positive"

# Tạo cột Label
df['Label'] = df['Rating'].apply(to_sentiment)

# In thử vài dòng đầu và thống kê nhãn
print(df[['Review Text', 'Rating', 'Label']].head())
print("\nSố lượng nhãn:")
print(df['Label'].value_counts())

# Lưu lại file CSV đã xử lý
df[['Review Text', 'Label']].to_csv("data/reviews_23k_clean.csv", index=False)
