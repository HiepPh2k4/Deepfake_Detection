import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Hàm trích xuất sample_id từ image_path
def extract_sample_id(image_path):
    filename = image_path.split('/')[-1]
    return filename.split('.')[0]

# Hàm đọc và kết hợp dữ liệu
def load_and_combine_predictions(face_file, voice_file, face_weight=0.6, voice_weight=0.4):
    try:
        face_df = pd.read_csv(face_file)
        voice_df = pd.read_csv(voice_file)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Không tìm thấy file: {e}")

    if face_df.empty or voice_df.empty:
        raise ValueError("Một trong hai file CSV rỗng hoặc không hợp lệ")

    voice_df['sample_id'] = voice_df['image_path'].apply(extract_sample_id)

    merged_df = pd.merge(
        face_df[['sample_id', 'label', 'face_prob_avg']],
        voice_df[['sample_id', 'true_label', 'probability_fake']],
        left_on=['sample_id', 'label'],
        right_on=['sample_id', 'true_label'],
        how='inner'
    )

    if merged_df.empty:
        raise ValueError("Không có dữ liệu khớp giữa hai file CSV")

    # Tính xác suất tổng hợp
    merged_df['final_prob'] = (face_weight * merged_df['face_prob_avg'] +
                               voice_weight * merged_df['probability_fake'])

    # Phân loại dựa trên ngưỡng 0.5
    merged_df['predicted_label'] = merged_df['final_prob'].apply(lambda x: 'fake' if x > 0.5 else 'real')

    # Chuyển true_label thành nhị phân để tính AUC
    merged_df['true_label_binary'] = merged_df['true_label'].map({'fake': 1, 'real': 0})

    # Đổi tên cột trong merged_df
    merged_df = merged_df.rename(columns={'face_prob_avg': 'face_prob', 'probability_fake': 'audio_prob'})

    return merged_df

# Hàm tính các chỉ số đánh giá
def evaluate_predictions(df):
    y_true = df['true_label']
    y_pred = df['predicted_label']
    y_true_binary = df['true_label_binary']
    y_score = df['final_prob']

    # Tính accuracy
    acc = accuracy_score(y_true, y_pred)

    # Tính F1-score
    f1 = f1_score(y_true, y_pred, pos_label='fake')

    # Tính AUC
    auc = roc_auc_score(y_true_binary, y_score)

    # Tính confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=['real', 'fake'])

    # Tính ROC curve
    fpr, tpr, _ = roc_curve(y_true_binary, y_score)

    return acc, f1, auc, cm, fpr, tpr

# Hàm tìm các trường hợp đặc biệt
def find_special_cases(df, threshold=0.5):
    # Trường hợp face real, voice fake, final real
    face_real_voice_fake_final_real = df[
        (df['face_prob'] <= threshold) &
        (df['audio_prob'] > threshold) &
        (df['final_prob'] <= threshold)
    ][['sample_id', 'true_label', 'face_prob', 'audio_prob', 'final_prob']]

    # Trường hợp face real, voice fake, final fake
    face_real_voice_fake_final_fake = df[
        (df['face_prob'] <= threshold) &
        (df['audio_prob'] > threshold) &
        (df['final_prob'] > threshold)
    ][['sample_id', 'true_label', 'face_prob', 'audio_prob', 'final_prob']]

    # Trường hợp face fake, voice real, final real
    face_fake_voice_real_final_real = df[
        (df['face_prob'] > threshold) &
        (df['audio_prob'] <= threshold) &
        (df['final_prob'] <= threshold)
    ][['sample_id', 'true_label', 'face_prob', 'audio_prob', 'final_prob']]

    # Trường hợp face fake, voice real, final fake
    face_fake_voice_real_final_fake = df[
        (df['face_prob'] > threshold) &
        (df['audio_prob'] <= threshold) &
        (df['final_prob'] > threshold)
    ][['sample_id', 'true_label', 'face_prob', 'audio_prob', 'final_prob']]

    return (face_real_voice_fake_final_real, face_real_voice_fake_final_fake,
            face_fake_voice_real_final_real, face_fake_voice_real_final_fake)

# Hàm lưu kết quả
def save_results(df, acc, f1, auc, cm, fpr, tpr, output_file, special_cases_file,
                 face_real_voice_fake_final_real, face_real_voice_fake_final_fake,
                 face_fake_voice_real_final_real, face_fake_voice_real_final_fake):
    # Tạo thư mục đầu ra nếu chưa tồn tại
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Lưu kết quả chính với cột đã đổi tên
    result_df = df[['sample_id', 'true_label', 'face_prob', 'audio_prob', 'final_prob', 'predicted_label']]
    result_df.to_csv(output_file, index=False)

    # Lưu các trường hợp đặc biệt
    special_cases = pd.concat([
        face_real_voice_fake_final_real.assign(case='Face Real Voice Fake Final Real'),
        face_real_voice_fake_final_fake.assign(case='Face Real Voice Fake Final Fake'),
        face_fake_voice_real_final_real.assign(case='Face Fake Voice Real Final Real'),
        face_fake_voice_real_final_fake.assign(case='Face Fake Voice Real Final Fake')
    ], ignore_index=True)
    special_cases.to_csv(special_cases_file, index=False)

    # Lưu confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['real', 'fake'], yticklabels=['real', 'fake'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(os.path.dirname(output_file), 'confusion_matrix.png'))
    plt.close()

    # Lưu ROC curve
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(os.path.dirname(output_file), 'roc_curve.png'))
    plt.close()

    print(f"Kết quả đã được lưu vào {output_file}")
    print(f"Các trường hợp đặc biệt đã được lưu vào {special_cases_file}")
    print(f"Confusion matrix đã được lưu vào {os.path.dirname(output_file)}/confusion_matrix.png")
    print(f"ROC curve đã được lưu vào {os.path.dirname(output_file)}/roc_curve.png")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")

# Hàm chính
def main():
    face_file = "output/output_face/aggregated_face_probs.csv"
    voice_file = "output/output_audio/all_predictions.csv"
    output_file = "output/output_multimodal/final_predictions_with_metrics.csv"
    special_cases_file = "output/output_multimodal/special_cases.csv"

    # Kết hợp dữ liệu và tính xác suất tổng hợp
    merged_df = load_and_combine_predictions(face_file, voice_file, face_weight=0.55, voice_weight=0.45)

    # Đánh giá
    acc, f1, auc, cm, fpr, tpr = evaluate_predictions(merged_df)

    # Tìm các trường hợp đặc biệt
    (face_real_voice_fake_final_real, face_real_voice_fake_final_fake,
     face_fake_voice_real_final_real, face_fake_voice_real_final_fake) = find_special_cases(merged_df)

    # Lưu kết quả
    save_results(merged_df, acc, f1, auc, cm, fpr, tpr, output_file, special_cases_file,
                 face_real_voice_fake_final_real, face_real_voice_fake_final_fake,
                 face_fake_voice_real_final_real, face_fake_voice_real_final_fake)

    print("\nConfusion Matrix:")
    print(cm)
    print("\nFace Real Voice Fake Final Real cases:")
    print(face_real_voice_fake_final_real)
    print("\nFace Real Voice Fake Final Fake cases:")
    print(face_real_voice_fake_final_fake)
    print("\nFace Fake Voice Real Final Real cases:")
    print(face_fake_voice_real_final_real)
    print("\nFace Fake Voice Real Final Fake cases:")
    print(face_fake_voice_real_final_fake)

    return (merged_df, acc, f1, auc, cm,
            face_real_voice_fake_final_real, face_real_voice_fake_final_fake,
            face_fake_voice_real_final_real, face_fake_voice_real_final_fake)

if __name__ == "__main__":
    (result_df, acc, f1, auc, cm,
     face_real_voice_fake_final_real, face_real_voice_fake_final_fake,
     face_fake_voice_real_final_real, face_fake_voice_real_final_fake) = main()