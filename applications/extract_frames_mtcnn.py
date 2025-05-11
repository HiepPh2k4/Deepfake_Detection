import cv2
import os
import shutil
from multiprocessing import Pool
import multiprocessing as mp
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Đường dẫn đến thư mục chứa video và nơi lưu khung hình
faceforensics_real_dir = "D:/Deepfake_Detection_project/data/FaceForensics1/original_sequences/actors/c23/videos/"
faceforensics_fake_dir = "D:/Deepfake_Detection_project/data/FaceForensics1/manipulated_sequences/DeepFakeDetection/c23/videos/"
output_dir = "D:/Deepfake_Detection_project/frames_mtcnn/"

# Tạo thư mục lưu khung hình nếu chưa có
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Đã tạo thư mục: {output_dir}")

# Biến toàn cục để lưu kết quả phát hiện khuôn mặt
detection_results = {
    'y_true': [],  # Nhãn thực tế: 1 nếu khung hình được cho là có khuôn mặt, 0 nếu không
    'y_pred': []   # Nhãn dự đoán: 1 nếu phát hiện khuôn mặt, 0 nếu không
}

# Hàm trích xuất và cắt khuôn mặt từ một video
def extract_frames(video_path, output_folder, label, max_frames=10, face_size=(48, 48)):
    # Khởi tạo MTCNN detector trong từng process
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
            # Chuyển sang RGB vì MTCNN yêu cầu
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Phát hiện khuôn mặt
            faces = detector.detect_faces(image_rgb)

            # Giả sử khung hình được chọn có khuôn mặt (y_true = 1)
            detection_results['y_true'].append(1)
            if len(faces) > 0:
                # Lấy khuôn mặt đầu tiên
                x, y, w, h = faces[0]['box']
                x, y = max(x, 0), max(y, 0)  # Đảm bảo không âm
                face_img = image[y:y+h, x:x+w]
                # Resize và chuyển sang ảnh xám
                face_img = cv2.resize(face_img, face_size)
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

                # Lưu khuôn mặt
                frame_name = f"{label}_{os.path.basename(video_path).split('.')[0]}_frame{count}.jpg"
                output_path = os.path.join(output_folder, frame_name)
                cv2.imwrite(output_path, face_img)
                print(f"Đã lưu khuôn mặt: {output_path}")
                detection_results['y_pred'].append(1)  # Phát hiện khuôn mặt
                count += 1
            else:
                print(f"Khung hình {frame_idx} không chứa khuôn mặt, bỏ qua.")
                detection_results['y_pred'].append(0)  # Không phát hiện khuôn mặt

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

    # Tính confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # Tính các metrics
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
    video_path, label, output_folder = args
    extract_frames(video_path, output_folder, label)

# Hàm chọn và trích xuất từ một thư mục với multiprocessing
def process_videos(video_dir, label, output_folder):
    if not os.path.exists(video_dir):
        print(f"Thư mục không tồn tại: {video_dir}")
        return

    # Lấy danh sách tất cả video .mp4
    videos = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    print(f"Tìm thấy {len(videos)} video trong thư mục {video_dir}")

    # Tạo danh sách tham số cho mỗi video
    video_args = [(os.path.join(video_dir, video), label, output_folder) for video in videos]

    # Sử dụng multiprocessing để xử lý song song với tqdm
    with Pool() as pool:
        for _ in tqdm(pool.imap(process_video, video_args), total=len(video_args), desc=f"Processing {label} videos"):
            pass

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    # Trích xuất từ FaceForensics++ (tất cả video thật và giả)
    process_videos(faceforensics_real_dir, "real", output_dir)
    process_videos(faceforensics_fake_dir, "fake", output_dir)

    # Tính và in confusion matrix cùng các metrics
    calculate_metrics()

    print("Hoàn tất trích xuất khuôn mặt và tính toán metrics!")