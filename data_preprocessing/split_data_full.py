import pandas as pd
from sklearn.model_selection import train_test_split

# Đường dẫn file nhãn
label_path = 'D:/Deepfake_Detection_project/data_preprocessing/output_split/label_split_full/label_rgb.csv'

# Đường dẫn đầu ra
train_path = 'D:/Deepfake_Detection_project/data_preprocessing/output_split/label_split_full/train.csv'
val_path = 'D:/Deepfake_Detection_project/data_preprocessing/output_split/label_split_full/val.csv'
test_path = 'D:/Deepfake_Detection_project/data_preprocessing/output_split/label_split_full/test.csv'

# Đọc file nhãn
df = pd.read_csv(label_path)

# Kiểm tra cột 'image_path'
if 'image_path' not in df.columns:
    raise KeyError("Cột 'image_path' không tồn tại.")

# Thêm cột 'method' từ image_path
df['method'] = df['image_path'].apply(lambda x: x.split('_')[1].lower() if len(x.split('_')) > 1 else 'unknown')

# In các cột và phương pháp
print("Các cột:", df.columns.tolist())
print("Phương pháp:", df['method'].unique())

# Khởi tạo danh sách lưu DataFrame
train_dfs, val_dfs, test_dfs = [], [], []

# Chia dữ liệu cho từng phương pháp
for method in df['method'].unique():
    group = df[df['method'] == method]
    train, temp = train_test_split(group, test_size=0.3, stratify=group['label'], random_state=42)
    val, test = train_test_split(temp, test_size=0.5, stratify=temp['label'], random_state=42)
    train_dfs.append(train)
    val_dfs.append(val)
    test_dfs.append(test)

# Gộp DataFrame
train_df = pd.concat(train_dfs)
val_df = pd.concat(val_dfs)
test_df = pd.concat(test_dfs)

# Lưu vào CSV
train_df[['image_path', 'label']].to_csv(train_path, index=False)
val_df[['image_path', 'label']].to_csv(val_path, index=False)
test_df[['image_path', 'label']].to_csv(test_path, index=False)

# In kết quả
print("\nĐã chia dữ liệu thành công!")
print(f"Train: {len(train_df)} khung hình (~{len(train_df)/len(df)*100:.1f}%)")
print(f"Validation: {len(val_df)} khung hình (~{len(val_df)/len(df)*100:.1f}%)")
print(f"Test: {len(test_df)} khung hình (~{len(test_df)/len(df)*100:.1f}%)")
print(f"File saved: {train_path}, {val_path}, {test_path}")