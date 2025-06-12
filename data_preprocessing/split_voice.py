import os
import shutil
import hashlib
from sklearn.model_selection import train_test_split
from tqdm import tqdm

DATA_DIR = "D:/Deepfake_Detection_project/data_preprocessing/frames_audio"
OUTPUT_DIR = "D:/Deepfake_Detection_project/data_preprocessing/output_audio"
TEMP_DIR = "D:/Deepfake_Detection_project/data_preprocessing/temp"
os.makedirs(TEMP_DIR, exist_ok=True)

def compute_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def main():
    # Step 1: Collect all files and label them based on higher-level directory
    all_files = []
    labels = []
    file_hashes = []
    for root, _, files in os.walk(DATA_DIR):
        print(f"Scanning directory: {root}")
        for file in files:
            if file.endswith(".png"):
                file_path = os.path.join(root, file)
                parent_dirs = root.split(os.sep)
                category_dir = next((d for d in parent_dirs if "RealAudio" in d or "FakeAudio" in d), None)
                if category_dir:
                    label = 1 if "FakeAudio" in category_dir else 0 if "RealAudio" in category_dir else None
                    if label is not None:
                        all_files.append(file_path)
                        labels.append(label)
                        file_hashes.append(compute_md5(file_path))
                    else:
                        print(f"Skipping file {file_path} due to unclear label in {category_dir}")
                else:
                    print(f"No valid category found for {file_path}")

    if not all_files:
        print("Không tìm thấy file nào trong thư mục frames_audio!")
        exit()
    print(f"Tổng số file: {len(all_files)}")

    # Step 2: Detect duplicates
    duplicates = {}
    for file_path, hash_val in zip(all_files, file_hashes):
        if hash_val not in duplicates:
            duplicates[hash_val] = []
        duplicates[hash_val].append(file_path)

    # Step 3: Remove duplicates and keep unique files
    unique_files = []
    unique_labels = []
    seen_hashes = set()
    for file_path, label, hash_val in zip(all_files, labels, file_hashes):
        if hash_val not in seen_hashes:
            unique_files.append(file_path)
            unique_labels.append(label)
            seen_hashes.add(hash_val)

    print(f"Số file duy nhất sau khi loại bỏ trùng lặp: {len(unique_files)}")

    # Step 4: Split into train/val/test with exact sizes
    total = len(unique_files)
    train_size = 5489  # Exact number for 70%
    val_size = 1176    # Exact number for 15%
    test_size = 1177   # Exact number for 15%, to total 7842

    # First split: train and (val + test)
    train_files, temp_files, train_labels, temp_labels = train_test_split(
        unique_files, unique_labels, train_size=train_size, stratify=unique_labels, random_state=42
    )

    # Second split: val and test from temp
    val_files, test_files, val_labels, test_labels = train_test_split(
        temp_files, temp_labels, train_size=val_size, stratify=temp_labels, random_state=42
    )

    # Verify split sizes
    print(f"Train files: {len(train_files)}")
    print(f"Val files: {len(val_files)}")
    print(f"Test files: {len(test_files)}")
    print(f"Total split: {len(train_files) + len(val_files) + len(test_files)}")

    # Step 5: Reorganize files into new directories with unique names
    for split in ["train", "validation", "test"]:
        split_dir = os.path.join(OUTPUT_DIR, split)
        shutil.rmtree(split_dir, ignore_errors=True)
        for label in ["fake", "real"]:
            label_dir = os.path.join(split_dir, label)
            os.makedirs(label_dir, exist_ok=True)

    copied_count = 0
    for file_path, label in tqdm(zip(unique_files, unique_labels), total=len(unique_files)):
        label_str = "fake" if label == 1 else "real"
        if file_path in train_files:
            split_dir = "train"
        elif file_path in val_files:
            split_dir = "validation"
        elif file_path in test_files:
            split_dir = "test"
        else:
            print(f"File not assigned: {file_path}")
            continue
        dst_dir = os.path.join(OUTPUT_DIR, split_dir, label_str)
        # Create unique file name based on original path
        relative_path = os.path.relpath(file_path, DATA_DIR)
        flat_name = os.path.join(*relative_path.split(os.sep)[1:]).replace(os.sep, "_") + ".png"
        dst_path = os.path.join(dst_dir, flat_name)
        try:
            shutil.copy(file_path, dst_path)
            copied_count += 1
        except Exception as e:
            print(f"Error copying {file_path} to {dst_path}: {e}")

    print(f"Total copied: {copied_count}")

    # Step 6: Clean up temporary directory
    shutil.rmtree(TEMP_DIR, ignore_errors=True)

    # Step 7: Verify splits
    total_samples = len(unique_files)
    for split in ["train", "validation", "test"]:
        split_dir = os.path.join(OUTPUT_DIR, split)
        fake_count = len([f for f in os.listdir(os.path.join(split_dir, "fake")) if f.endswith(".png")])
        real_count = len([f for f in os.listdir(os.path.join(split_dir, "real")) if f.endswith(".png")])
        total = fake_count + real_count
        ratio = (total / total_samples) * 100 if total_samples > 0 else 0
        print(f"{split}: Fake={fake_count}, Real={real_count}, Total={total}, Ratio={ratio:.1f}%")

if __name__ == "__main__":
    main()