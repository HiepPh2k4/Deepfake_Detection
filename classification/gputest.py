# import tensorflow as tf
#
# print("Phiên bản TensorFlow:", tf.__version__)
# print("GPU Available:", tf.config.list_physical_devices('GPU'))
# print("CUDA Available:", tf.test.is_built_with_cuda())
# print("CUDA Toolkit version: Kiểm tra bằng 'nvcc --version' (nên là 11.8)")


import torch

print("Phiên bản PyTorch:", torch.__version__)
print("GPU Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Số lượng GPU:", torch.cuda.device_count())
    print("Tên GPU:", torch.cuda.get_device_name(0))
    print("CUDA Version:", torch.version.cuda)
else:
    print("Không tìm thấy GPU hoặc CUDA chưa được cấu hình đúng.")