import pandas as pd

# Đọc file 1000 dòng đã tách
df = pd.read_csv("data/reviews_1000_from_23k.csv")

# Hàm ánh xạ Rating -> Label
def to_sentiment(rating):
    if rating in [1, 2]:
        return "Negative"
    elif rating == 3:
        return "Neutral"
    else:
        return "Positive"

# Tạo cột Label
df["Label"] = df["Rating"].apply(to_sentiment)

# Giữ lại 2 cột cần thiết
df_clean = df[["Review Text", "Label"]]

# Lưu lại file để train
df_clean.to_csv("data/reviews_1000_clean.csv", index=False)

print("File reviews_1000_clean.csv đã sẵn sàng!")
