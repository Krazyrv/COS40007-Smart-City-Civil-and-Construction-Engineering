from pathlib import Path
from PIL import Image
import random, re, json, shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

# defining directories
ROOT = Path("dataset")
MERGED = Path("merged_dataset")
IMG_EXTS = {".png"}

# config for balancing not_rubbish images with rubbish images
ADD_NOT_RUBBISH_BALANCED = True
# not_rubbish:rubbish ratio
MAX_NOT_RUBBISH_PER_POSITIVE = 0.1  # 1 here means 1:1 ratio

# train/val split ratio
TRAIN_RATIO = 0.8

# random seed for deterministic splits
random.seed(42)

# creating merged directories
(MERGED / "images" / "train").mkdir(parents=True, exist_ok=True)
(MERGED / "images" / "val").mkdir(parents=True, exist_ok=True)
(MERGED / "labels" / "train").mkdir(parents=True, exist_ok=True)
(MERGED / "labels" / "val").mkdir(parents=True, exist_ok=True)

print("setup success")


# labelme to YOLO conversion helpers
def unique_name(teammate_name: str, orig_name: str) -> str:
    """
    create a unique, filesystem-friendly base name: teammate_origbasename (no extension)
    prevents file overwriting
    """
    base = Path(orig_name).stem
    # using regex sanitize teammate name and base to avoid spaces/slashes
    t = re.sub(r"[^a-zA-Z0-9_-]+", "_", teammate_name.strip())
    b = re.sub(r"[^a-zA-Z0-9_-]+", "_", base)
    return f"{t}_{b}"


def bbox_from_points(points):
    """
    LabelMe: shape.points can be 2 points (rect) or a polygon list
    return(xmin, ymin, xmax, ymax)
    """

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def to_yolo_line(cls_id, bbox, img_w, img_h):
    """
    YOLO labels must be normalized to [0, -1] and in center-width-height format.
    """
    xmin, ymin, xmax, ymax = bbox

    # "clamp" coords to image bounds (no negatives/out-of-range)
    xmin, ymin = max(0, xmin), max(0, ymin)
    xmax, ymax = min(img_w, xmax), min(img_h, ymax)

    # validate box dimensions (must have positive width and height)
    if xmax <= xmin or ymax <= ymin:
        return None  # Skip invalid boxes

    # convert to YOLO (normalized)
    xc = ((xmin + xmax) / 2) / img_w
    yc = ((ymin + ymax) / 2) / img_h
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h

    # return in a format the YOLO expects
    return f"{cls_id} {xc} {yc} {w} {h}"


def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS


# finding teammates folders, collect files
teammates = [d for d in ROOT.iterdir() if d.is_dir()]
print("teammate folders:", [t.name for t in teammates])

#### adding support for both JSON and XML annotations


# parse json annotations function
def parse_labelme_json(jp):
    """Parse LabelMe JSON -> unified dict."""
    data = json.loads(jp.read_text())
    # ensure required keys exist
    data.setdefault("imagePath", None)
    data.setdefault("imageWidth", data.get("imageWidth"))
    data.setdefault("imageHeight", data.get("imageHeight"))
    # LabelMe already has shapes with 'points'
    # we only keep rectangle/polygon by extracting 2-point rectangles for downstream
    unified_shapes = []
    for sh in data.get("shapes", []):
        lab = sh.get("label")
        pts = sh.get("points", [])
        if not lab or not pts:
            continue

        # normalize to 2-point rectangle [[xmin, ymin], [xmax, ymax]]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        rect = [[float(min(xs)), float(min(ys))], [float(max(xs)), float(max(ys))]]
        unified_shapes.append({"label": lab, "points": rect})

    return {
        "imagePath": data.get("imagePath"),
        "imageWidth": data.get("imageWidth"),
        "imageHeight": data.get("imageHeight"),
        "shapes": unified_shapes,
    }


# parse XML function
def parse_voc_xml(xp):
    """Parse Pascal VOC XML -> unified dict."""
    root = ET.fromstring(xp.read_text())

    # filenmae
    fn_node = root.find("filename")
    image_path = fn_node.text.strip() if fn_node is not None else None

    # size
    sz = root.find("size")
    w = h = None
    if sz is not None:
        try:
            w = int(sz.find("width").text)
            h = int(sz.find("height").text)
        except Exception:
            pass

    # objects -> shapes
    unified_shapes = []
    for obj in root.findall("object"):
        name_node = obj.find("name")
        lab = name_node.text.strip() if name_node is not None else None
        bb = obj.find("bndbox")
        if lab and bb is not None:
            try:
                xmin = float(bb.find("xmin").text)
                ymin = float(bb.find("ymin").text)
                xmax = float(bb.find("xmax").text)
                ymax = float(bb.find("ymax").text)
            except Exception:
                continue

            # normalize to the same format as LabelMe: two-point rect
            rect = [[xmin, ymin], [xmax, ymax]]
            unified_shapes.append({"label": lab, "points": rect})

    return {
        "imagePath": image_path,  # may need stem matching fallback later
        "imageWidth": w,
        "imageHeight": h,
        "shapes": unified_shapes,
    }


# Collect JSONs and build a per-teammate index by (imagePath or image stem)

json_entries = []  # list of (teammate_name, json_path, data)
classes_set = set()  # dynamically collect the unique class names (e.g. mattress, cans, bottles, couch, toys).


for tdir in teammates:
    # for each teammate directoy
    # look for an 'annotations' subfolder where LabelMe JSONs live.
    ann_dir = tdir / "annotations"
    if not ann_dir.exists():
        # if a teammate hasn't provide annotations yet, skip
        continue

    # iterate over all .json and .xml files in that annotations folder
    for ap in list(ann_dir.glob("*.json")) + list(ann_dir.glob("*.xml")):
        try:
            # read and parse the JSON file into a Python dict
            if ap.suffix.lower() == ".json":
                data = parse_labelme_json(ap)
            else:  # ".xml"
                data = parse_voc_xml(ap)
        except Exception as e:
            # if the JSON is malformed or unreadable, don't crash the pipeline
            # print a help message and skip the file
            print(f"Skipping bad annotation files: {ap} ({e})")
            continue

        # store the trio (who/where/what) so later steps can:
        # - find the corresponding image in that teammate's 'rubbush/' folder
        # - convert shapes -> YOLO labels
        json_entries.append((tdir.name, ap, data))

# canonical classes + alias map and a normalizer
CANONICAL = [
    "mattress",
    "electrical_goods",
    "couch",
    "trolley",
    "toy",
    "clothes",
    "cardboard",
    "rubbish_bag",
    "furniture",
    "litter",
    "carton",
    "aluminium_cans",
    "bottle",
]

# handcrafted aliases (typos and common name mismatches)
ALIASES = {
    # spacing/hyphens/underscores variants
    "rubbish bag": "rubbish_bag",
    # common mislabels / synonyms
    "stray_trolley": "trolley",
    "scrap": "litter",
    "jug": "bottle",
    "garbage": "rubbish_bag",
    "furniture_scraps": "furniture",
    "chair": "furniture",
    "glass_bottle": "bottle",
    "plastic_bottle": "bottle",
    "torn_paper": "litter",
}

# pre-compute a lowercase lookup for canonical names
CANON_LOWER = {c.lower(): c for c in CANONICAL}


def canonicalize(raw_label: str, fuzzy=True):
    """
    Convert raw teammate label -> canonical class name or None if unknown.
    Strat:
        1) normalize spaces/hyphens/underscores, lowercase
        2) apply ALIASES
        3) exact match to CANONICAL
    """

    if not raw_label:
        return None

    s = raw_label.strip().lower()
    s = re.sub(r"[\s\-]+", "_", s)  # spaces/hyphens -> underscore
    s = re.sub(r"__+", "_", s)  # collapse doubles

    # direct alias
    if s in ALIASES:
        s = ALIASES[s]

    # exact canonical match
    if s in CANON_LOWER:
        return CANON_LOWER[s]

    return None


# collect canonicalized class labels from all annotation files
for tname, ap, data in json_entries:
    for sh in data.get("shapes", []):
        lab = sh.get("label")
        if lab:
            # apply canonicalization before adding to classes set
            canon = canonicalize(lab)
            if canon:
                # add to a set to ensure each class appears only one overall.
                classes_set.add(canon)

# sort the unordered set so class indices are stable and reproducible
CLASSES = sorted(classes_set)

# diagnostics
print(f"\nDiscovered {len(CLASSES)} canonical classes:", CLASSES)
print(f"Total annotation files discovered: {len(json_entries)}")


normalized_counts = Counter()
unknown_examples = defaultdict(int)

total_shapes_before = 0
total_shapes_after = 0

for i, (tname, ap, data) in enumerate(json_entries):
    new_shapes = []
    for sh in data.get("shapes", []):
        total_shapes_before += 1
        raw = sh.get("label")
        pts = sh.get("points")
        if not raw or not pts:
            continue

        canon = canonicalize(raw)
        if canon is None:
            # keep a small log to fix upstream labels later
            key = f"{raw} -> ? (in {ap.name})"
            unknown_examples[key] += 1
            continue

        normalized_counts[canon] += 1
        new_shapes.append({"label": canon, "points": sh["points"]})

    data["shapes"] = new_shapes
    json_entries[i] = (tname, ap, data)

    total_shapes_after += len(new_shapes)

print("normalization complete")
print("shapes before:", total_shapes_before, "after:", total_shapes_after)
print("\nclass counts after normalization:")
for cls in CANONICAL:
    print(f"{cls:16s} {normalized_counts[cls]}")

if unknown_examples:
    print("\n unknown/mismatched labels (top 15):")
    for k, v in list(sorted(unknown_examples.items(), key=lambda x: -x[1])):
        print(f" {v:4d} {k}")

else:
    print("\n all good")

# merge annotated images (rubbish with JSON), then add balanced not_rubbish

labels_written = 0
images_copied = 0

# book-keeping:
merged_list = []  # all merged image paths
pos_image_ids = set()  # stems of positive (rubbish) images after unique-naming
neg_image_ids = set()  # stems of negative (not_rubbish) images after unique-naming


def find_rubbish_image(teammate_dir: Path, image_filename: str):
    """
    given a teammate's root folder and an annotation's image filename (imagePath),
    try to find the *actual* image in teammate/rubbish/.
    1) first attempt exact match (same name + extension)
    2) if not found, fall back to matching by stem (ignore extension), so 'foo.jpg' can match 'foo.PNG'
    """
    rub_dir = teammate_dir / "rubbish"

    # exact filename match
    if not rub_dir.exists():
        return None
    cand = rub_dir / image_filename
    if cand.exists():
        return cand

    # fallback: match by stem (any extension)
    stem = Path(image_filename).stem
    for p in rub_dir.iterdir():
        if is_image_file(p) and p.stem == stem:
            return p
    return None


# merge ONLY annotated rubbish images ----
for teammate_name, jp, data in json_entries:
    # prefer labelmel's recorded image name; if missing try same name as JSON but '.jpg'
    image_fn = (
        data.get("imagePath") or jp.with_suffix(".jpg").name
    )  # heuristic fallback
    teammate_dir = ROOT / teammate_name

    # find the matching rubbish image file for this annotation JSON
    img_path = find_rubbish_image(teammate_dir, image_fn)
    if img_path is None:
        # if we can't find the image, skip without error
        print(f"No matching rubbish image for JSON: {jp.name} (expected {image_fn})")
        continue

    # open to get size for normalization when writing to YOLO labels
    try:
        with Image.open(img_path) as im:
            w, h = im.size
    except Exception as e:
        # don't let a corrupt image through
        print(f"Cannot open image {img_path}: {e}")
        continue

    # a unique base name to avoid collisions between teammates
    ub = unique_name(teammate_name, img_path.name)  # unique base
    out_img = MERGED / "images" / f"{ub}{img_path.suffix.lower()}"

    # copy the image into the merged dataset if we haven't yet
    if not out_img.exists():
        shutil.copy2(img_path, out_img)
        images_copied += 1
        merged_list.append(out_img)
        pos_image_ids.add(out_img.stem)  # remember this positive's stem

    # write YOLO label
    out_lbl = MERGED / "labels" / f"{ub}.txt"
    with open(out_lbl, "a") as f:
        # each shape in LabelMe is a labled region
        for sh in data.get("shapes", []):
            lab = sh.get("label")
            pts = sh.get("points", [])
            # skip if label is empty, no points, or label not in our discovered class list
            if not lab or not pts or lab not in CLASSES:
                continue

            # map class name -> numeric class id via CLASSES ordering
            cls_id = CLASSES.index(lab)

            # convert polygon/rect points into a tight bounding box
            xmin, ymin, xmax, ymax = bbox_from_points(pts)

            # convert that box into YOLO normalized "class xc, yc, w, h" line
            line = to_yolo_line(cls_id, (xmin, ymin, xmax, ymax), w, h)
            if line:
                f.write(line + "\n")
                labels_written += 1


# diagnose
num_positives = len(pos_image_ids)
print(f"Positive (annotated rubbish) images merged: {num_positives}")
print(f"YOLO label lines written: {labels_written}")

# ddd balanced not_rubbish (empty labels) ----
if ADD_NOT_RUBBISH_BALANCED:
    # Gather all available not_rubbish images across teammates
    pool = []
    for tdir in [d for d in ROOT.iterdir() if d.is_dir()]:
        nr_dir = tdir / "not_rubbish"
        if not nr_dir.exists():
            continue
        for p in nr_dir.iterdir():
            if is_image_file(p):
                pool.append((tdir.name, p))
    print(f"Found {len(pool)} candidate not_rubbish images across teammates.")

    need = int(num_positives * MAX_NOT_RUBBISH_PER_POSITIVE)
    if need <= 0:
        print("No negatives requested (need <= 0).")
    else:
        # Sample without replacement (cap at available)
        random.shuffle(pool)
        take = min(need, len(pool))
        sampled = pool[:take]

        added_nr = 0
        for teammate_name, p in sampled:
            # build a unique name for the negative image too
            ub = unique_name(teammate_name, p.name)
            out_img = MERGED / "images" / f"{ub}{p.suffix.lower()}"

            # if by change a positive already used this unique ame, skip
            if out_img.exists():  # avoid collision with a positive of same ub
                continue
            shutil.copy2(p, out_img)
            open(MERGED / "labels" / f"{ub}.txt", "w").close()  # empty label
            neg_image_ids.add(out_img.stem)
            merged_list.append(out_img)
            added_nr += 1

        print(f"Added not_rubbish (negatives): {added_nr} (requested {need})")

# save classes and data.yaml (YOYO-ready)
# classes.txt

(CLASS_TXT := MERGED / "classes.txt").write_text("\n".join(CLASSES))
print("wrote", CLASS_TXT)

# data.yaml for YOLO/Ultralytics
yaml_text = f"""# Auto-generated data.yaml
path: {MERGED.resolve()}
train: images/train
val: images/val

names:
"""

for i, name in enumerate(CLASSES):
    yaml_text += f" {i}: {name}\n"

(DATA_YAML := MERGED / "data.yaml").write_text(yaml_text)
print("wrote", DATA_YAML)

# split merged images into train/val and move pairs

all_imgs = sorted((MERGED / "images").glob("*.*"))
random.shuffle(all_imgs)

split_idx = int(TRAIN_RATIO * len(all_imgs))
train_imgs = all_imgs[:split_idx]
val_imgs = all_imgs[split_idx:]


def move_pair(img_path: Path, phase: str):
    dst_img = MERGED / "images" / phase / img_path.name
    lbl_src = MERGED / "labels" / (img_path.stem + ".txt")
    dst_lbl = MERGED / "labels" / phase / (img_path.stem + ".txt")
    shutil.move(str(img_path), str(dst_img))
    if lbl_src.exists():
        shutil.move(str(lbl_src), str(dst_lbl))


for p in train_imgs:
    move_pair(p, "train")
for p in val_imgs:
    move_pair(p, "val")

print(f"train images: {len(train_imgs)}")
print(f"val images: {len(val_imgs)}")
