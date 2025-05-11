import cv2
import os
import shutil
from multiprocessing import Pool
from tqdm import tqdm

# Đường dẫn
faceforensics_real_dir = "D:/Deepfake_Detection_project/data/FaceForensics1/original_sequences/actors/c23/videos/"
faceforensics_fake_dir = "D:/Deepfake_Detection_project/data/FaceForensics1/manipulated_sequences/DeepFakeDetection/c23/videos/"
output_dir = "D:/Deepfake_Detection_project/frames_haar_cascades/"

# Kiểm tra thư mục
for dir_path in [faceforensics_real_dir, faceforensics_fake_dir]:
    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"Thư mục không tồn tại: {dir_path}")

# Tạo thư mục lưu khung hình
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Đã tạo thư mục: {output_dir}")

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
if face_cascade.empty():
    raise FileNotFoundError("Không tìm thấy file haarcascade_frontalface_default.xml!")

# Hàm trích xuất khuôn mặt
def extract_frames(video_path, output_folder, label, max_frames=10, face_size=(48, 48)):
    print(f"Đang xử lý video: {video_path}")
    vidcap = cv2.VideoCapture(video_path)
    if not vidcap.isOpened():
        print(f"Không thể mở video: {video_path}")
        return

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
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                face_img = image[y:y+h, x:x+w]
                face_img = cv2.resize(face_img, face_size)
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

                frame_name = f"{label}_{os.path.basename(video_path).split('.')[0]}_frame{count}.jpg"
                output_path = os.path.join(output_folder, frame_name)
                if os.path.exists(output_path):
                    print(f"File {output_path} đã tồn tại, bỏ qua!")
                    continue
                cv2.imwrite(output_path, face_img)
                print(f"Đã lưu khuôn mặt: {output_path}")
                count += 1
            else:
                print(f"Khung hình {frame_idx} không chứa khuôn mặt, bỏ qua.")

        success, image = vidcap.read()
        frame_idx += 1
    if count == 0:
        print(f"Video {video_path} không trích xuất được khuôn mặt nào!")
    vidcap.release()

# Hàm phụ cho multiprocessing
def process_video(args):
    video_path, label, output_folder = args
    extract_frames(video_path, output_folder, label)

# Hàm xử lý video
def process_videos(video_dir, label, output_folder):
    if not os.path.exists(video_dir):
        print(f"Thư mục không tồn tại: {video_dir}")
        return

    videos = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    print(f"Tìm thấy {len(videos)} video trong thư mục {video_dir}")
    video_args = [(os.path.join(video_dir, video), label, output_folder) for video in videos]

    with Pool(processes=os.cpu_count()) as pool:
        list(tqdm(pool.imap(process_video, video_args), total=len(video_args)))

# Chạy chính
if __name__ == '__main__':
    process_videos(faceforensics_real_dir, "real", output_dir)
    process_videos(faceforensics_fake_dir, "fake", output_dir)

print("Hoàn tất trích xuất khuôn mặt!")
