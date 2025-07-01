import os
for split in ['train', 'validation', 'test']:
    for label in ['real', 'fake']:
        path = f'D:/Deepfake_Detection_project/data_preprocessing/voice_data/{split}/{label}/'
        print(f'{split}/{label}: {len(os.listdir(path))} files')