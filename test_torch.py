import torch

print("PyTorch version:", torch.__version__)

if torch.backends.mps.is_available():
    print("Apple GPU acceleration is available!")
    device = torch.device("mps")
else:
    print("MPS unavailable. We can use CPU.")
    device = torch.device("cpu")

print("Selected device:", device)

x = torch.ones(3, device=device)
print("Test calculation:", x * 2)

