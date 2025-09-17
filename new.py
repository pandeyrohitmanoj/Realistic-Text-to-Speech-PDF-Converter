import torch
def reset_cuda_context():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Force context destruction and recreation
        torch._C._cuda_clearCublasWorkspaces()
        torch.cuda.synchronize()
        # Initialize fresh context
        dummy = torch.tensor([1.0]).cuda()
        del dummy
        torch.cuda.empty_cache()
        
reset_cuda_context()