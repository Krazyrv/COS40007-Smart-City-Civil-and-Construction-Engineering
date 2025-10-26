#!/usr/bin/env python3
"""
Fix label normalization in dataset by applying aliases to raw annotation files.
This ensures consistent class names across all teammates' data.
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
import re
import shutil

# Same aliases as in run_data.py
ALIASES = {
    # spacing/hyphens/underscores variants
    "rubbish bag": "rubbish_bag",
    # common mislabels / synonyms
    "stray_trolley": "trolley",
    "scrap": "litter",
    "jug": "carton",
    "garbage": "rubbish_bag",
    "furniture_scraps": "furniture",
    "chair": "furniture",
    "glass_bottle": "bottle",
    "plastic_bottle": "bottle",
    "torn_paper": "litter",
    "bottle": "bottle",
    "can": "aluminium_cans",
    "paper": "litter",
    "cardboard": "cardboard",
    "plastic": "litter",
    "glass": "bottle",
    "metal": "aluminium_cans",
}

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
    "blanket",
    "bottle",
]

CANON_LOWER = {c.lower(): c for c in CANONICAL}


def normalize_label(raw_label: str):
    """Normalize a label using the same logic as run_data.py"""
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


def fix_json_file(json_path):
    """Fix labels in a JSON annotation file"""
    try:
        with open(json_path, "r") as f:
            data = json.load(f)

        changed = False
        for shape in data.get("shapes", []):
            if "label" in shape:
                original_label = shape["label"]
                normalized_label = normalize_label(original_label)

                if normalized_label and normalized_label != original_label:
                    print(f"  {original_label} → {normalized_label}")
                    shape["label"] = normalized_label
                    changed = True

        if changed:
            # Backup original file
            backup_path = json_path.with_suffix(json_path.suffix + ".backup")
            shutil.copy2(json_path, backup_path)

            # Write fixed file
            with open(json_path, "w") as f:
                json.dump(data, f, indent=2)

            print(f"✅ Fixed {json_path.name}")
            return True

    except Exception as e:
        print(f"❌ Error processing {json_path.name}: {e}")

    return False


def fix_xml_file(xml_path):
    """Fix labels in an XML annotation file"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        changed = False
        for obj in root.findall("object"):
            name_elem = obj.find("name")
            if name_elem is not None:
                original_label = name_elem.text.strip()
                normalized_label = normalize_label(original_label)

                if normalized_label and normalized_label != original_label:
                    print(f"  {original_label} → {normalized_label}")
                    name_elem.text = normalized_label
                    changed = True

        if changed:
            # Backup original file
            backup_path = xml_path.with_suffix(xml_path.suffix + ".backup")
            shutil.copy2(xml_path, backup_path)

            # Write fixed file
            tree.write(xml_path, encoding="utf-8", xml_declaration=True)

            print(f"✅ Fixed {xml_path.name}")
            return True

    except Exception as e:
        print(f"❌ Error processing {xml_path.name}: {e}")

    return False


def fix_teammate_data(teammate_dir):
    """Fix all annotation files for a teammate"""
    annotations_dir = teammate_dir / "annotations"

    if not annotations_dir.exists():
        print(f"⚠️  No annotations folder found for {teammate_dir.name}")
        return

    print(f"\n🔧 Fixing labels for {teammate_dir.name}...")

    fixed_count = 0

    # Fix JSON files
    for json_file in annotations_dir.glob("*.json"):
        if fix_json_file(json_file):
            fixed_count += 1

    # Fix XML files
    for xml_file in annotations_dir.glob("*.xml"):
        if fix_xml_file(xml_file):
            fixed_count += 1

    print(f"📊 Fixed {fixed_count} files for {teammate_dir.name}")


def main():
    """Fix all teammate annotation files"""
    dataset_dir = Path("dataset")

    if not dataset_dir.exists():
        print("❌ Dataset directory not found!")
        return

    teammates = [d for d in dataset_dir.iterdir() if d.is_dir()]

    print(f"Starting label normalization for {len(teammates)} teammates...")
    print(f"Teammates: {[t.name for t in teammates]}")

    total_fixed = 0
    for teammate_dir in teammates:
        annotations_dir = teammate_dir / "annotations"
        if annotations_dir.exists():
            fix_teammate_data(teammate_dir)

    print(f"\nLabel normalization complete!")


if __name__ == "__main__":
    main()
