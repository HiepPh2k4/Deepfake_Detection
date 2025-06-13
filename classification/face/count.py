import pandas as pd
import numpy as np


def calculate_pos_weight(csv_path):
    """Tính pos_weight tối ưu từ dữ liệu training"""

    # Đọc dữ liệu
    data = pd.read_csv(csv_path)

    # Đếm số lượng
    real_count = len(data[data['label'] == 'real'])
    fake_count = len(data[data['label'] == 'fake'])
    total = len(data)

    # Tính tỉ lệ
    real_ratio = real_count / total * 100
    fake_ratio = fake_count / total * 100

    # Tính pos_weight (cho class 1 = fake)
    pos_weight = real_count / fake_count

    print("=" * 50)
    print("PHÂN TÍCH DỮ LIỆU")
    print("=" * 50)
    print(f"Tổng số ảnh: {total:,}")
    print(f"Số ảnh REAL: {real_count:,} ({real_ratio:.1f}%)")
    print(f"Số ảnh FAKE: {fake_count:,} ({fake_ratio:.1f}%)")
    print(f"Tỉ lệ Real:Fake = {pos_weight:.2f}:1")
    print()

    print("=" * 50)
    print("KHUYẾN NGHỊ POS_WEIGHT")
    print("=" * 50)
    print(f"pos_weight optimal = {pos_weight:.2f}")
    print(f"pos_weight hiện tại = 5.91")

    if abs(pos_weight - 5.91) < 0.1:
        print("✅ Giá trị hiện tại GẦN ĐÚNG")
    else:
        print("❌ NÊN SỬA lại pos_weight")
        print(f"   Thay đổi từ 5.91 → {pos_weight:.2f}")

    print()
    print("=" * 50)
    print("CÁC OPTION KHÁC")
    print("=" * 50)

    # Option 1: Balanced weight
    balanced_weight = np.sqrt(pos_weight)
    print(f"1. Balanced (căn bậc 2): {balanced_weight:.2f}")

    # Option 2: Conservative
    conservative_weight = pos_weight * 0.7
    print(f"2. Conservative (70%): {conservative_weight:.2f}")

    # Option 3: Aggressive
    aggressive_weight = pos_weight * 1.3
    print(f"3. Aggressive (130%): {aggressive_weight:.2f}")

    return pos_weight


# Sử dụng:
# THAY ĐỔI PATH NÀY THÀNH PATH THỰC TẾ CỦA BẠN
csv_path = "G:/Hiep/Deepfake_Detection/data_preprocessing/output_split/label_full_remote/train_rgb.csv"

try:
    optimal_pos_weight = calculate_pos_weight(csv_path)
except FileNotFoundError:
    print("❌ Không tìm thấy file CSV!")
    print("Hãy thay đổi csv_path thành đường dẫn thực tế của bạn")
except Exception as e:
    print(f"❌ Lỗi: {e}")



# print("-----------------------------------------------------------------------")
#
# import os
# fake_count = len(os.listdir("D:/Deepfake_Detection_project/data_preprocessing/output_audio/train/fake"))
# real_count = len(os.listdir("D:/Deepfake_Detection_project/data_preprocessing/output_audio/train/real"))
# print(f"Fake: {fake_count}, Real: {real_count}")
