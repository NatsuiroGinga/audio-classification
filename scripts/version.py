import onnxruntime
import torch
import torch.version

print(f"{onnxruntime.__version__=}")
print(f"{torch.__version__=}")
print(f"{torch.cuda.is_available()=}")
print(f"{torch.version.cuda=}")
print(f"{torch.backends.cudnn.version()=}")
