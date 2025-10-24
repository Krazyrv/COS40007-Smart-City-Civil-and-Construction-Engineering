import torch
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from .frcnn_trainer import YoloDetectionDataset, collate_detection
from torch.utils.data import DataLoader
from torchvision import transforms as T
from model.config.settings import NUM_CLASSES

class SSDTrainer:
    def __init__(self, img_dirs, lbl_dirs, num_classes=10, hp=None, device="cpu"):
        self.device = torch.device(device)
        size = (hp or {}).get("imgsz", 320)
        self.train_ds = YoloDetectionDataset(img_dirs["train"], lbl_dirs["train"], size=size, num_classes=NUM_CLASSES)
        self.val_ds   = YoloDetectionDataset(img_dirs["val"],   lbl_dirs["val"],   size=size, num_classes=NUM_CLASSES)
        self.train_dl = DataLoader(self.train_ds, batch_size=(hp or {}).get("batch",8), shuffle=True,  num_workers=0, collate_fn=collate_detection)
        self.val_dl   = DataLoader(self.val_ds,   batch_size=(hp or {}).get("batch",8), shuffle=False, num_workers=0, collate_fn=collate_detection)
        self.model = ssdlite320_mobilenet_v3_large(weights_backbone="DEFAULT", num_classes=1+NUM_CLASSES).to(self.device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=(hp or {}).get("lr",2e-4), weight_decay=(hp or {}).get("weight_decay",5e-4))
        self.epochs = (hp or {}).get("epochs",10)

    def train(self):
        for ep in range(self.epochs):
            self.model.train(); total=0.0
            for imgs, targets in self.train_dl:
                imgs=[x.to(self.device) for x in imgs]
                targets=[{k:v.to(self.device) for k,v in t.items()} for t in targets]
                loss=sum(self.model(imgs, targets).values())
                self.opt.zero_grad(); loss.backward(); self.opt.step()
                total += loss.item()
            print(f"[SSD] Epoch {ep+1}/{self.epochs} Loss {total/len(self.train_dl):.4f}")
        return {"status":"ok"}

    def validate(self): return {"val_images": len(self.val_ds)}
    def predict(self, sources): raise NotImplementedError  # similar to Faster R-CNN
    def save(self, path): torch.save(self.model.state_dict(), path)
    def load(self, path): self.model.load_state_dict(torch.load(path, map_location=self.device)); return self
