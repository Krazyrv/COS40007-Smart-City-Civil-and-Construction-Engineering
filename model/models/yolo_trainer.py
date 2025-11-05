from ultralytics import YOLO
from pathlib import Path
from model.core.base import Trainer
import torch
import torch.nn as nn

class YOLOTrainer(Trainer):
    def __init__(self, data_yaml: Path, ckpt="yolov8n.pt", project="runs_models/yolo", name="yolo", hp=None, device="gpu"):
        self.data_yaml = str(data_yaml)
        if name == "yolov10n":
            print ("Using YOLOv10n checkpoint")
            ckpt = "yolov10n.pt"
        else:
            print ("Using YOLOv8n checkpoint")
        self.model = YOLO(ckpt)
        self.project, self.name = f"{project}/{name}", name
        self.hp = dict(epochs=50, 
                        imgsz=640, 
                        batch=16,  
                        lr0=2e-3, 
                        workers=0,
                        # hsv_h=0, hsv_s=0, hsv_v=0, weight_decay=0,mosaic=0, degrees = 0.0, patience=12, warmup_epochs=0,
                        # optimizer="Adam"
                        weight_decay=5e-4,
                        mosaic=0.1, 
                        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, 
                        degrees = 30.0,
                        flipud=0.5,   # vertical flip probability (up-down)
                        fliplr=0.5,   # horizontal flip probability (left-right)
                        patience=50, 
                        warmup_epochs=3,
                        optimizer="SGD", # or "SGD", "Adam", "AdamW", "RMSProp"
                        class_weights=[0.0196, 0.2, 0.0384, 0.0555, 0.0243, 0.00625, 0.0256,
                        0.2, 0.00578, 0.04, 0.0714, 0.0303, 0.00885, 0.01449,
                        0.3333, 0.0129, 0.00361, 0.0909, 0.0270, 1.0, 0.0909, 0.00478]
                       )
        if hp: self.hp.update(hp)
        self.device = device

    def train(self):
        # Customize the loss function to include class weights
        # from ultralytics.utils import loss
        # loss.BCEWithLogitsLoss = lambda *args, **kwargs: nn.BCEWithLogitsLoss(weight=self.hp["class_weights"], reduction="mean")
        print(f"Train with {self.hp['epochs']} epochs")
        r = self.model.train(
            data=self.data_yaml,
            epochs=self.hp["epochs"], imgsz=self.hp["imgsz"], batch=self.hp["batch"],
            workers=self.hp["workers"], optimizer=self.hp["optimizer"], lr0=self.hp["lr0"], weight_decay=self.hp["weight_decay"],
            mosaic=self.hp["mosaic"], hsv_h=self.hp["hsv_h"], hsv_s=self.hp["hsv_s"], hsv_v=self.hp["hsv_v"],
            patience=self.hp["patience"],warmup_epochs= self.hp['warmup_epochs'], degrees = self.hp['degrees'],
            project=self.project, name=self.name, device=self.device
        )
        return r

    def validate(self):
        return self.model.val(data=self.data_yaml, imgsz=self.hp["imgsz"], project=f"{self.project}/{self.name}", name=f"{self.name}/{self.name}_val", device=self.device)

    def predict(self, sources):
        return self.model.predict(source=sources, imgsz=self.hp["imgsz"], project=self.project, name=f"{self.name}_pred", device=self.device, save=True)

    def save(self, path):  # best.pt is already saved by Ultralytics
        # optional extra export
        self.model.export(format="onnx", dynamic=True)

    def load(self, path):
        self.model = YOLO(str(path))
        return self
