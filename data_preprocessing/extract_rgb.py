import cv2
import os
import shutil
from multiprocessing import Pool
import multiprocessing as mp
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Đường dẫn đến thư mục cha chứa dữ liệu FaceForensics
base_data_dir = "D:/Deepfake_Detection_project/data/FaceForensics_c23/"
output_dir = "D:/Deepfake_Detection_project/data_preprocessing/frames_mtcnn_rgb_500/"

# Danh sách thư mục cần bỏ qua
excluded_folders = ["actors", "DeepFakeDetection"]

# Tạo thư mục lưu khung hình nếu chưa có
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Đã tạo thư mục: {output_dir}")

# Biến toàn cục để lưu kết quả phát hiện khuôn mặt
detection_results = {
    'y_true': [],
    'y_pred': []
}
frame_metadata = []  # Lưu thông tin khung hình

# Hàm trích xuất và cắt khuôn mặt từ một video
def extract_frames(video_path, output_folder, label, source_folder_name, max_frames=10, target_size=(299, 299)):
    from mtcnn import MTCNN
    detector = MTCNN()

    print(f"Đang xử lý video: {video_path}")
    vidcap = cv2.VideoCapture(video_path)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        print(f"Video {video_path} không đọc được hoặc rỗng.")
        return

    step = max(total_frames // max_frames, 1)
    count = 0
    frame_idx = 0
    success, image = vidcap.read()

    while success and count < max_frames:
        if frame_idx % step == 0:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            faces = detector.detect_faces(image_rgb)

            detection_results['y_true'].append(1)
            if len(faces) > 0:
                x, y, w, h = faces[0]['box']
                x, y = max(x, 0), max(y, 0)

                # Mở rộng vùng cắt 1.3 lần
                expansion_factor = 1.3
                new_w = int(w * expansion_factor)
                new_h = int(h * expansion_factor)
                center_x = x + w // 2
                center_y = y + h // 2
                new_x = max(center_x - new_w // 2, 0)
                new_y = max(center_y - new_h // 2, 0)
                new_x_end = min(new_x + new_w, image.shape[1])
                new_y_end = min(new_y + new_h, image.shape[0])
                new_x = max(new_x_end - new_w, 0)
                new_y = max(new_y_end - new_h, 0)

                # Cắt vùng khuôn mặt đã mở rộng
                face_img = image[new_y:new_y_end, new_x:new_x_end]

                # Resize về kích thước 299x299 (RGB)
                face_img = cv2.resize(face_img, target_size)

                # Lưu khung hình dưới dạng RGB
                frame_name = f"{label}_{source_folder_name}_{os.path.basename(video_path).split('.')[0]}_frame{count}.jpg"
                output_path = os.path.join(output_folder, frame_name)
                cv2.imwrite(output_path, face_img)
                print(f"Đã lưu khung hình: {output_path}")

                # Lưu thông tin khung hình vào metadata
                frame_metadata.append({
                    "image_path": output_path,
                    "label": label
                })

                detection_results['y_pred'].append(1)
                count += 1
            else:
                print(f"Khung hình {frame_idx} không chứa khuôn mặt, bỏ qua.")
                detection_results['y_pred'].append(0)

        success, image = vidcap.read()
        frame_idx += 1
    vidcap.release()

# Hàm tính confusion matrix và các metrics
def calculate_metrics():
    y_true = detection_results['y_true']
    y_pred = detection_results['y_pred']

    if len(y_true) == 0 or len(y_pred) == 0:
        print("Không có dữ liệu để tính metrics.")
        return

    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print("\nMetrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

# Hàm phụ để gọi extract_frames với multiprocessing
def process_video(args):
    video_path, label, output_folder, source_folder_name = args
    extract_frames(video_path, output_folder, label, source_folder_name)

# Hàm chọn và trích xuất từ một thư mục với multiprocessing
def process_videos(video_dir, label, output_folder):
    if not os.path.exists(video_dir):
        print(f"Thư mục không tồn tại: {video_dir}")
        return

    # Lấy tên thư mục nguồn (Deepfakes, FaceShifter, youtube, v.v.)
    source_folder_name = os.path.basename(os.path.dirname(os.path.dirname(video_dir)))

    videos = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    print(f"Tìm thấy {len(videos)} video trong thư mục {video_dir}")

    video_args = [(os.path.join(video_dir, video), label, output_folder, source_folder_name) for video in videos]

    with Pool() as pool:
        for _ in tqdm(pool.imap(process_video, video_args), total=len(video_args), desc=f"Processing {label} videos"):
            pass

# Hàm xử lý video thật từ original_sequences, bỏ qua thư mục actors
def process_real_folders(base_data_dir, output_dir):
    original_dir = os.path.join(base_data_dir, "original_sequences")

    if os.path.exists(original_dir):
        original_subdirs = [d for d in os.listdir(original_dir)
                            if os.path.isdir(os.path.join(original_dir, d)) and d not in excluded_folders]

        for subdir in original_subdirs:
            video_dir = os.path.join(original_dir, subdir, "c23", "videos")
            if os.path.exists(video_dir):
                print(f"\nBắt đầu xử lý video thật từ: {video_dir}")
                process_videos(video_dir, "real", output_dir)
            else:
                print(f"Thư mục không tồn tại: {video_dir}")
    else:
        print(f"Thư mục video thật không tồn tại: {original_dir}")

# Hàm xử lý video giả từ manipulated_sequences, bỏ qua DeepFakeDetection
def process_fake_folders(base_data_dir, output_dir):
    manipulated_dir = os.path.join(base_data_dir, "manipulated_sequences")

    if os.path.exists(manipulated_dir):
        manipulated_subdirs = [d for d in os.listdir(manipulated_dir)
                               if os.path.isdir(os.path.join(manipulated_dir, d)) and d not in excluded_folders]

        for subdir in manipulated_subdirs:
            video_dir = os.path.join(manipulated_dir, subdir, "c23", "videos")
            if os.path.exists(video_dir):
                print(f"\nBắt đầu xử lý video giả từ: {video_dir}")
                process_videos(video_dir, "fake", output_dir)
            else:
                print(f"Thư mục không tồn tại: {video_dir}")
    else:
        print(f"Thư mục video giả không tồn tại: {manipulated_dir}")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    # Gọi hàm để xử lý tất cả thư mục thật
    process_real_folders(base_data_dir, output_dir)

    # Gọi hàm để xử lý tất cả thư mục giả
    process_fake_folders(base_data_dir, output_dir)

    # Tính và in metrics
    calculate_metrics()

    print("Hoàn tất trích xuất khung hình và tính toán metrics từ các thư mục video!")