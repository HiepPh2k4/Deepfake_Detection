import cv2
import os
import shutil

# Đường dẫn đến thư mục chứa video và nơi lưu khung hình
faceforensics_real_dir = "D:/project/data/FaceForensics/manipulated_sequences/DeepFakeDetection/c40/videos/"
faceforensics_fake_dir = "D:/project/data/FaceForensics/original_sequences/actors/c40/videos/"
output_dir = "D:/project/frames/"

# Tạo thư mục lưu khung hình nếu chưa có
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Đã tạo thư mục: {output_dir}")


# Hàm trích xuất khung hình từ một video
def extract_frames(video_path, output_folder, label, max_frames=10):
    print(f"Đang xử lý video: {video_path}")
    vidcap = cv2.VideoCapture(video_path)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        print(f"Video {video_path} không đọc được hoặc rỗng.")
        return

    # Tính bước nhảy để lấy max_frames khung hình đều nhau
    step = max(total_frames // max_frames, 1)
    count = 0
    frame_idx = 0
    success, image = vidcap.read()

    while success and count < max_frames:
        if frame_idx % step == 0:
            frame_name = f"{label}_{os.path.basename(video_path).split('.')[0]}_frame{count}.jpg"
            output_path = os.path.join(output_folder, frame_name)
            cv2.imwrite(output_path, image)
            print(f"Đã lưu khung hình: {output_path}")
            count += 1
        success, image = vidcap.read()
        frame_idx += 1
    vidcap.release()


# Hàm chọn và trích xuất từ một thư mục
def process_videos(video_dir, label, output_folder, num_videos=150):
    if not os.path.exists(video_dir):
        print(f"Thư mục không tồn tại: {video_dir}")
        return

    # Lấy danh sách video (chỉ lấy 150 video đầu tiên)
    videos = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    videos = videos[:num_videos]  # Chọn 150 video
    print(f"Tìm thấy {len(videos)} video trong thư mục {video_dir}")

    for video in videos:
        video_path = os.path.join(video_dir, video)
        extract_frames(video_path, output_folder, label)


# Trích xuất từ FaceForensics++ (150 thật, 150 giả)
process_videos(faceforensics_real_dir, "real", output_dir)
process_videos(faceforensics_fake_dir, "fake", output_dir)

print("Hoàn tất trích xuất khung hình!")