import torch, numpy, pandas
print("numpy:", numpy.__version__)
print("pandas:", pandas.__version__)
print("torch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
