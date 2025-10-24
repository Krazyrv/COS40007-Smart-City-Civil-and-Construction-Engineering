from pathlib import Path

DATA_ROOT = Path("merged_dataset")
DATA_YAML = DATA_ROOT / "data.yaml"
IMAGES = {
    "train": DATA_ROOT / "images" / "train",
    "val":   DATA_ROOT / "images" / "val",
}
LABELS = {
    "train": DATA_ROOT / "labels" / "train",
    "val":   DATA_ROOT / "labels" / "val",
}

# common hyperparams
HP = dict(
    epochs=50,
    imgsz=640,
    batch=16,
    lr=2e-3,
    weight_decay=5e-4,
    workers=4,          # safer on macOS
    patience=12,
)

NUM_CLASSES = 10  # your object classes (not counting background)
