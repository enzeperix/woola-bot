import torch

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA Device Count: {torch.cuda.device_count()}")
    print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")  # Assuming you have one GPU

    # Simple tensor operation on GPU
    x = torch.rand(5, 3).cuda()
    y = torch.rand(5, 3).cuda()
    z = x + y
    print(z)