from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

IMG_EXTS = {".jpg",".png"}

class PresenceDataset(Dataset):
    """Binary classifier dataset: rubbish present? (label file empty→0, non-empty→1)."""
    def __init__(self, img_dir, lbl_dir, size=384):
        self.img_dir, self.lbl_dir = Path(img_dir), Path(lbl_dir)
        self.images = sorted([p for p in self.img_dir.iterdir() if p.suffix.lower() in IMG_EXTS])
        self.tfms = T.Compose([T.Resize((size,size)), T.ToTensor()])

    def __len__(self): return len(self.images)
    def __getitem__(self, i):
        p = self.images[i]
        y = 0
        lf = self.lbl_dir / (p.stem + ".txt")
        if lf.exists() and lf.read_text().strip(): y = 1
        x = Image.open(p).convert("RGB")
        return self.tfms(x), torch.tensor(y, dtype=torch.long)

def collate_detection(batch):
    return tuple(zip(*batch))
