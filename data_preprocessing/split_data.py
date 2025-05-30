import pandas as pd
from sklearn.model_selection import train_test_split

# Đường dẫn đến file nhãn
label_path = 'D:/Deepfake_Detection_project/data_preprocessing/output_split/label_split_full/label_rgb.csv'

# Đường dẫn đầu ra cho các file train, validation, test
train_path = 'D:/Deepfake_Detection_project/data_preprocessing/output_split/label_split_pp/train.csv'
val_path = 'D:/Deepfake_Detection_project/data_preprocessing/output_split/label_split_pp/val.csv'
test_path = 'D:/Deepfake_Detection_project/data_preprocessing/output_split/label_split_pp/test.csv'

# Đọc file nhãn
df = pd.read_csv(label_path)

# In các cột để kiểm tra
print("Các cột trong file nhãn:", df.columns.tolist())

# Kiểm tra cột 'image_path'
if 'image_path' not in df.columns:
    raise KeyError("Cột 'image_path' không tồn tại trong file nhãn. Vui lòng kiểm tra tên cột.")


# Thêm cột 'method' dựa trên image_path
def extract_method(image_path):
    parts = image_path.split('_')
    if len(parts) < 2:
        return 'unknown'
    # Lấy phần method (phần thứ 2 sau dấu '_')
    return parts[1].lower()


df['method'] = df['image_path'].apply(extract_method)

# In các phương pháp trước khi lọc
print("Các phương pháp trước khi lọc:", df['method'].unique())
print("Số lượng khung hình theo phương pháp trước khi lọc:\n", df['method'].value_counts())

# Loại trừ các khung hình có image_path chứa DeepFakeDetection, actors, FaceShifter
excluded_patterns = ['deepfakedetection', 'actors', 'faceshifter']
excluded_df = df[df['image_path'].str.lower().str.contains('|'.join(excluded_patterns), na=False)]
df = df[~df['image_path'].str.lower().str.contains('|'.join(excluded_patterns), na=False)]

# Báo cáo số lượng khung hình bị loại bỏ
print(f"\nSố khung hình bị loại bỏ: {len(excluded_df)}")
if len(excluded_df) > 0:
    print("Số lượng khung hình bị loại bỏ theo method:\n", excluded_df['method'].value_counts())

# Kiểm tra số lượng khung hình sau khi loại trừ
print(f"\nTổng số khung hình sau khi loại trừ: {len(df)}")
if len(df) == 0:
    raise ValueError(
        "Không còn khung hình nào sau khi loại trừ. Dữ liệu có thể chỉ chứa DeepFakeDetection, actors, hoặc FaceShifter. Kiểm tra extract_rgb.py và label.py.")

print("Số lượng khung hình theo phương pháp sau khi lọc:\n", df['method'].value_counts())

# Lấy danh sách các phương pháp duy nhất
methods = df['method'].unique()

# Khởi tạo danh sách để lưu các DataFrame train, val, test
train_dfs = []
val_dfs = []
test_dfs = []

# Chia dữ liệu cho từng phương pháp
for method in methods:
    group = df[df['method'] == method]
    print(f"\nChia dữ liệu cho phương pháp: {method} ({len(group)} khung hình)")

    # Chia thành train và temp (70:30)
    train, temp = train_test_split(group, test_size=0.3, stratify=group['label'], random_state=42)
    # Chia temp thành val và test (50:50 của temp, tức 15:15 của tổng)
    val, test = train_test_split(temp, test_size=0.5, stratify=temp['label'], random_state=42)

    train_dfs.append(train)
    val_dfs.append(val)
    test_dfs.append(test)

    print(f"Train: {len(train)} khung hình, Validation: {len(val)} khung hình, Test: {len(test)} khung hình")

# Gộp các DataFrame
train_df = pd.concat(train_dfs)
val_df = pd.concat(val_dfs)
test_df = pd.concat(test_dfs)

# Lưu vào file CSV (chỉ giữ cột 'image_path' và 'label')
train_df[['image_path', 'label']].to_csv(train_path, index=False)
val_df[['image_path', 'label']].to_csv(val_path, index=False)
test_df[['image_path', 'label']].to_csv(test_path, index=False)

# In thông tin cuối cùng
print("\nĐã chia dữ liệu thành công!")
print(f"Tổng số khung hình train: {len(train_df)} (~{len(train_df) / len(df) * 100:.1f}%)")
print(f"Tổng số khung hình validation: {len(val_df)} (~{len(val_df) / len(df) * 100:.1f}%)")
print(f"Tổng số khung hình test: {len(test_df)} (~{len(test_df) / len(df) * 100:.1f}%)")
print(f"Đã tạo các file: {train_path}, {val_path}, {test_path}")