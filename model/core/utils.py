import random, numpy as np, torch, time

def set_seed(seed=42):
    random.seed(seed);
    np.random.seed(seed);
    torch.manual_seed(seed)


def get_device(pref="auto"):
    if pref == "cpu": return torch.device("cpu")
    if torch.cuda.is_available(): return torch.device("cuda:0")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

class Timer:
    def __enter__(self): self.t=time.time(); return self
    def __exit__(self, *a): self.elapsed=time.time()-self.t