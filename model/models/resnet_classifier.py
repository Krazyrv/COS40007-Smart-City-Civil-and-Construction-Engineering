import torch
from torchvision import models, transforms as T
from torch.utils.data import DataLoader
from PIL import Image
from model.core.base import Trainer
from model.core.data_readers import PresenceDataset

class ResNetClassifier(Trainer):
    def __init__(self, img_dirs, lbl_dirs, hp=None, device="cpu"):
        self.device = torch.device(device)
        size = (hp or {}).get("imgsz", 384)
        self.train_ds = PresenceDataset(img_dirs["train"], lbl_dirs["train"], size=size)
        self.val_ds   = PresenceDataset(img_dirs["val"],   lbl_dirs["val"],   size=size)
        self.train_dl = DataLoader(self.train_ds, batch_size=(hp or {}).get("batch",16), shuffle=True,  num_workers=0)
        self.val_dl   = DataLoader(self.val_ds,   batch_size=(hp or {}).get("batch",16), shuffle=False, num_workers=0)

        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 2)
        self.model = self.model.to(self.device)

        self.opt = torch.optim.AdamW(self.model.parameters(), lr=(hp or {}).get("lr",1e-4), weight_decay=(hp or {}).get("weight_decay",1e-4))
        self.crit = torch.nn.CrossEntropyLoss()
        self.epochs = (hp or {}).get("epochs",10)

    def train(self):
        for ep in range(self.epochs):
            self.model.train(); tot=0.0; correct=seen=0
            for x,y in self.train_dl:
                x,y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                loss = self.crit(logits, y)
                self.opt.zero_grad(); loss.backward(); self.opt.step()
                tot += loss.item()
                pred = logits.argmax(1)
                correct += (pred==y).sum().item(); seen += y.numel()
            print(f"[ResNet] Ep {ep+1}/{self.epochs} Loss {tot/len(self.train_dl):.4f} Acc {correct/seen:.3f}")
        return {"status":"ok"}

    def validate(self):
        self.model.eval(); correct=seen=0
        with torch.no_grad():
            for x,y in self.val_dl:
                x,y = x.to(self.device), y.to(self.device)
                pred = self.model(x).argmax(1)
                correct += (pred==y).sum().item(); seen += y.numel()
        acc = correct/max(1,seen)
        print(f"[ResNet] Val Acc {acc:.3f}")
        return {"val_acc": acc}

    def predict(self, sources):  # list of image paths
        out=[]
        tfm = T.Compose([T.Resize((384,384)), T.ToTensor()])
        self.model.eval()
        with torch.no_grad():
            for p in sources:
                x = tfm(Image.open(p).convert("RGB")).unsqueeze(0).to(self.device)
                pred = self.model(x).softmax(1)[0].tolist()
                out.append({"path": p, "probs": {"not_rubbish": pred[0], "rubbish": pred[1]}})
        return out

    def save(self, path): torch.save(self.model.state_dict(), path)
    def load(self, path): self.model.load_state_dict(torch.load(path, map_location=self.device)); return self
