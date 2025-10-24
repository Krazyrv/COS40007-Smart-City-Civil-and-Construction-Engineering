# ... existing imports ...
ALL_MODELS = ["yolov8n", "yolov10n", "frcnn", "ssd", "resnet50"]

def run_one(model_name, device):
    from model.config.settings import DATA_YAML, IMAGES, LABELS, HP
    from model.models.yolo_trainer import YOLOTrainer
    from model.models.frcnn_trainer import FasterRCNNTrainer
    from model.models.ssd_trainer import SSDTrainer
    from model.models.resnet_classifier import ResNetClassifier

    if model_name in {"yolov8n","yolov10n"}:
        ckpt = "yolov8n.pt" if model_name=="yolov8n" else "yolov10n.pt"
        trainer = YOLOTrainer(DATA_YAML, ckpt=ckpt, name=model_name, hp=HP, device=device)
    elif model_name == "frcnn":
        trainer = FasterRCNNTrainer(IMAGES, LABELS, num_classes=10, hp=HP, device=device)
    elif model_name == "ssd":
        trainer = SSDTrainer(IMAGES, LABELS, num_classes=10, hp=HP, device=device)
    else:
        trainer = ResNetClassifier(IMAGES, LABELS, hp=dict(epochs=15, batch=16, lr=1e-4), device=device)

    print(f"\n=== Training {model_name} on {device} ===")
    try:
        trainer.train()
        print("Validating …")
        metrics = trainer.validate()
        print("Metrics:", metrics)
    except Exception as e:
        print(f"⚠️ {model_name} failed: {e}")

def main():
    import argparse
    from model.core.utils import get_device, set_seed
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["yolov8n","yolov10n","frcnn","ssd","resnet50","all"], required=True)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    set_seed(42)
    device = str(get_device(args.device))

    if args.model == "all":
        for m in ALL_MODELS:
            run_one(m, device)
        print("\nCompleted all models.")
    else:
        run_one(args.model, device)

if __name__ == "__main__":
    main()
