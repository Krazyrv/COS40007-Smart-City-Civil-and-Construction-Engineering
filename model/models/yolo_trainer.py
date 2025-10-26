from ultralytics import YOLO
from pathlib import Path
from model.core.base import Trainer
import torch

class YOLOTrainer(Trainer):
    """
    Extended YOLO Trainer with improved augmentation, model selection, and auto GPU detection.
    """

    def __init__(self, 
                 data_yaml: Path,
                 ckpt="yolov8s.pt",            # upgraded default from nano → small for better accuracy
                 project="runs_models",
                 name="yolo",
                 hp=None,
                 device=None):
        
        # auto-detect device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.data_yaml = str(data_yaml)
        self.model = YOLO(ckpt)
        self.project, self.name = project, name

        # Default hyperparameters (tuned for mid-size YOLOv8s/v10s)
        self.hp = dict(
            epochs=100,
            imgsz=640,
            batch=16,
            workers=2,
            optimizer="AdamW",
            lr0=2e-3,
            weight_decay=5e-4,
            momentum=0.937,
            mosaic=1.0,          # stronger augmentations
            mixup=0.15,
            degrees=5.0,
            translate=0.1,
            scale=0.5,
            shear=2.0,
            flipud=0.2,
            fliplr=0.5,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            patience=15          # early stopping tolerance
        )
        if hp:
            self.hp.update(hp)

    # ===============================================
    # 🚀 Train with enhanced hyperparameters
    # ===============================================
    def train(self):
        print(f"🔧 Starting YOLO training on device: {self.device}")

        r = self.model.train(
            data=self.data_yaml,
            epochs=self.hp["epochs"],
            imgsz=self.hp["imgsz"],
            batch=self.hp["batch"],
            workers=self.hp["workers"],
            optimizer=self.hp["optimizer"],
            lr0=self.hp["lr0"],
            weight_decay=self.hp["weight_decay"],
            momentum=self.hp["momentum"],
            mosaic=self.hp["mosaic"],
            mixup=self.hp["mixup"],
            degrees=self.hp["degrees"],
            translate=self.hp["translate"],
            scale=self.hp["scale"],
            shear=self.hp["shear"],
            flipud=self.hp["flipud"],
            fliplr=self.hp["fliplr"],
            hsv_h=self.hp["hsv_h"],
            hsv_s=self.hp["hsv_s"],
            hsv_v=self.hp["hsv_v"],
            patience=self.hp["patience"],
            project=self.project,
            name=self.name,
            device=self.device,
            exist_ok=True,
            pretrained=True,     # leverage pre-trained weights
            verbose=True
        )

        print("✅ Training complete. Best weights saved automatically under runs_models/")
        return r

    # ===============================================
    # 🧪 Validation with metrics and plots
    # ===============================================
    def validate(self):
        print("📊 Running validation and generating performance plots...")
        results = self.model.val(
            data=self.data_yaml,
            imgsz=self.hp["imgsz"],
            project=self.project,
            name=f"{self.name}_val",
            device=self.device,
            save_json=True,     # produces COCO-format metrics
            plots=True           # saves confusion matrix, PR, F1 curves
        )
        print("✅ Validation complete. Plots and metrics saved in runs_models/")
        return results

    # ===============================================
    # 🔍 Prediction
    # ===============================================
    def predict(self, sources, conf=0.25):
        print(f"🔍 Running detection on {sources} with conf={conf}")
        return self.model.predict(
            source=sources,
            conf=conf,  # lower default threshold to catch more detections
            imgsz=self.hp["imgsz"],
            project=self.project,
            name=f"{self.name}_pred",
            device=self.device,
            save=True,
            exist_ok=True
        )

    # ===============================================
    # 💾 Save and Export
    # ===============================================
    def save(self, path=None):
        """
        Export model to multiple formats for deployment.
        """
        export_path = path or Path(self.project) / self.name / "exports"
        export_path.mkdir(parents=True, exist_ok=True)
        print(f"💾 Exporting model to {export_path}")

        # Export ONNX and TorchScript versions
        self.model.export(format="onnx", dynamic=True)
        self.model.export(format="torchscript")

        print("✅ Model exported in ONNX and TorchScript formats.")

    # ===============================================
    # 🔄 Load Model
    # ===============================================
    def load(self, path):
        print(f"🔄 Loading YOLO model from: {path}")
        self.model = YOLO(str(path))
        return self
