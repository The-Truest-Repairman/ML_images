import torch

def test_torch_cuda():
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA device count:", torch.cuda.device_count())
        print("CUDA device name:", torch.cuda.get_device_name(0))
        
        # Test a simple tensor operation on GPU
        x = torch.rand(5, 5, device=device)
        y = torch.rand(5, 5, device=device)
        z = x + y
        print("Sample tensor operation on GPU successful:")
        print(z)
    else:
        print("CUDA not available. Running on CPU instead.")
        x = torch.rand(5, 5)
        y = torch.rand(5, 5)
        z = x + y
        print("Sample tensor operation on CPU successful:")
        print(z)

if __name__ == "__main__":
    test_torch_cuda()