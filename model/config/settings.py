from pathlib import Path
# utils to read num classes from data.yaml
import yaml
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
    epochs=200,
    imgsz=640,
    batch=16,
    lr=2e-3,
    weight_decay=5e-4,
    workers=4,          # safer on macOS
    # patience=12,
)


def load_num_classes_from_yaml(data_yaml: Path) -> int:
    y = yaml.safe_load(Path(data_yaml).read_text())
    names = y.get("names")
    if isinstance(names, dict):
        return len(names)
    elif isinstance(names, list):
        return len(names)
    raise ValueError("Could not read 'names' from data.yaml")


NUM_CLASSES = load_num_classes_from_yaml(DATA_YAML) 





