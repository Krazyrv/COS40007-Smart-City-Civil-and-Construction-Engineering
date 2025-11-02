from ultralytics import YOLO
from pathlib import Path
from model.core.base import Trainer

class YOLOTrainer(Trainer):
    def __init__(self, data_yaml: Path, ckpt="yolov8n.pt", project="runs_models", name="yolo", hp=None, device="cpu"):
        self.data_yaml = str(data_yaml)
        self.model = YOLO(ckpt)
        self.project, self.name = project, name
        self.hp = dict(epochs=50, imgsz=640, batch=16, workers=0, lr0=2e-3, weight_decay=5e-4,
                       mosaic=0.1, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, patience=12, warmup_epochs=3, degrees = 30.0,
                       optimizer="SGD", # or "SGD", "Adam", "RMSProp"
                       )
        if hp: self.hp.update(hp)
        self.device = device

    def train(self):
        r = self.model.train(
            data=self.data_yaml,
            epochs=self.hp["epochs"], imgsz=self.hp["imgsz"], batch=self.hp["batch"],
            workers=self.hp["workers"], optimizer="AdamW", lr0=self.hp["lr0"], weight_decay=self.hp["weight_decay"],
            mosaic=self.hp["mosaic"], hsv_h=self.hp["hsv_h"], hsv_s=self.hp["hsv_s"], hsv_v=self.hp["hsv_v"],
            patience=self.hp["patience"],warmup_epochs= self.hp['warmup_epochs'], degrees = self.hp['degrees'],
            project=self.project, name=self.name, device=self.device,
        )
        return r

    def validate(self):
        return self.model.val(data=self.data_yaml, imgsz=self.hp["imgsz"], project=self.project, name=f"{self.name}_val", device=self.device)

    def predict(self, sources):
        return self.model.predict(source=sources, imgsz=self.hp["imgsz"], project=self.project, name=f"{self.name}_pred", device=self.device, save=True)

    def save(self, path):  # best.pt is already saved by Ultralytics
        # optional extra export
        self.model.export(format="onnx", dynamic=True)

    def load(self, path):
        self.model = YOLO(str(path))
        return self
