import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from model.core.base import Trainer
from model.core.data_readers import collate_detection, IMG_EXTS

class YoloDetectionDataset(Dataset):
    def __init__(self, img_dir, lbl_dir, size=640, num_classes=10):
        self.img_dir, self.lbl_dir = Path(img_dir), Path(lbl_dir)
        self.images = sorted([p for p in self.img_dir.iterdir() if p.suffix.lower() in IMG_EXTS])
        self.tfms = T.Compose([T.Resize((size,size)), T.ToTensor()])
        self.num_classes = num_classes

    def __len__(self): return len(self.images)

    def _read_yolo(self, lbl, w, h):
        boxes, labels = [], []
        if not lbl.exists(): return boxes, labels
        for line in lbl.read_text().splitlines():
            if not line.strip(): continue
            cls, xc, yc, bw, bh = line.split()
            cls = int(cls); xc, yc, bw, bh = map(float, (xc,yc,bw,bh))
            x = xc*w; y = yc*h; ww=bw*w; hh=bh*h
            boxes.append([x-ww/2, y-hh/2, x+ww/2, y+hh/2])
            labels.append(cls+1)  # background=0
        return boxes, labels

    def __getitem__(self, i):
        p = self.images[i]
        im = Image.open(p).convert("RGB"); w,h = im.size
        boxes, labels = self._read_yolo(self.lbl_dir/(p.stem+".txt"), w, h)
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([i]),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
        }
        return self.tfms(im), target

class FasterRCNNTrainer(Trainer):
    def __init__(self, img_dirs, lbl_dirs, num_classes=10, hp=None, device="cpu"):
        self.device = torch.device(device)
        size = (hp or {}).get("imgsz", 640)
        self.train_ds = YoloDetectionDataset(img_dirs["train"], lbl_dirs["train"], size=size, num_classes=num_classes)
        self.val_ds   = YoloDetectionDataset(img_dirs["val"],   lbl_dirs["val"],   size=size, num_classes=num_classes)
        self.train_dl = DataLoader(self.train_ds, batch_size=(hp or {}).get("batch",4), shuffle=True,  num_workers=0, collate_fn=collate_detection)
        self.val_dl   = DataLoader(self.val_ds,   batch_size=(hp or {}).get("batch",4), shuffle=False, num_workers=0, collate_fn=collate_detection)
        self.model = fasterrcnn_resnet50_fpn(weights="DEFAULT", num_classes=1+num_classes).to(self.device)
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
            print(f"[FRCNN] Epoch {ep+1}/{self.epochs} Loss {total/len(self.train_dl):.4f}")
        return {"status":"ok"}

    def validate(self):
        self.model.eval()
        # minimal placeholder (full mAP eval requires COCO evaluator)
        return {"val_images": len(self.val_ds)}

    def predict(self, sources):
        self.model.eval()
        outs=[]
        for p in sources:
            im = Image.open(p).convert("RGB")
            x = T.ToTensor()(im).to(self.device).unsqueeze(0)
            with torch.no_grad():
                out = self.model(x)[0]
            outs.append(out)
        return outs

    def save(self, path): torch.save(self.model.state_dict(), path)
    def load(self, path): self.model.load_state_dict(torch.load(path, map_location=self.device)); return self
