import os
import cv2
from mtcnn import MTCNN
from tqdm import tqdm
from multiprocessing import Pool

#  Config
INPUT_VIDEO_DIR = r"D:\Deepfake_Detection_project\data_preprocessing\fakeavceleb_audios_videos\video"
OUTPUT_FRAME_DIR = r"D:\Deepfake_Detection_project\data_preprocessing\fakeavceleb_audios_videos\video_frame_faces"
TARGET_SIZE = (299, 299)
FRAMES_PER_VIDEO = 3


def extract_faces_from_video(args):
    video_path, label = args
    from mtcnn import MTCNN
    detector = MTCNN()

    basename = os.path.splitext(os.path.basename(video_path))[0]
    output_subdir = os.path.join(OUTPUT_FRAME_DIR, label, basename)

    # 💡 Bỏ qua nếu đã đủ số frame
    if os.path.exists(output_subdir) and len(os.listdir(output_subdir)) >= FRAMES_PER_VIDEO:
        return

    os.makedirs(output_subdir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        return

    step = max(total_frames // FRAMES_PER_VIDEO, 1)
    count = 0
    frame_idx = 0
    success, frame = cap.read()

    while success and count < FRAMES_PER_VIDEO:
        if frame_idx % step == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = detector.detect_faces(rgb)
            if faces:
                x, y, w, h = faces[0]['box']
                x, y = max(x, 0), max(y, 0)

                new_w, new_h = int(w * 1.3), int(h * 1.3)
                cx, cy = x + w // 2, y + h // 2
                new_x = max(cx - new_w // 2, 0)
                new_y = max(cy - new_h // 2, 0)
                new_x_end = min(new_x + new_w, frame.shape[1])
                new_y_end = min(new_y + new_h, frame.shape[0])
                new_x = max(new_x_end - new_w, 0)
                new_y = max(new_y_end - new_h, 0)

                face_crop = frame[new_y:new_y_end, new_x:new_x_end]
                face_crop = cv2.resize(face_crop, TARGET_SIZE)

                out_path = os.path.join(output_subdir, f"frame{count}.jpg")
                cv2.imwrite(out_path, face_crop)
                count += 1
        success, frame = cap.read()
        frame_idx += 1
    cap.release()

def get_video_tasks():
    tasks = []
    for label in ["real", "fake"]:
        label_dir = os.path.join(INPUT_VIDEO_DIR, label)
        for file in os.listdir(label_dir):
            if file.endswith(".mp4"):
                full_path = os.path.join(label_dir, file)
                tasks.append((full_path, label))
    return tasks

def main():
    tasks = get_video_tasks()
    print(f"Tổng số video cần xử lý: {len(tasks)}")

    with Pool(processes=4) as pool:
        list(tqdm(pool.imap_unordered(extract_faces_from_video, tasks), total=len(tasks)))

    print("Trích xuất xong 3 frames/video với multiprocessing!")

if __name__ == "__main__":
    main()
