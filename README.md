# Smart-City-Civil-and-Construction-Engineering

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Environment Setup and Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Results](#results)
- [Contributors](#contributors)
- [License](#license)

## Overview

The system uses labeled image data to train machine learning models capable of distinguishing between:

- `rubbish`:
  - **mattress, electrical goods, chair, couch, trolley, toy, clothes, cartoon, rubbish bag, furniture**
- `not_rubbish`:
  - **what are not rubbish**
- `damage-sign3`:
  - damaged sign: **broken sheet, bent, crack, graffiti, rust/dust**
  - non-damaged sign: **just non-damaged sign ;)**

## Environment Setup:

- `conda create -n computer-vision python=3.10 -y`
- `pip install requirements.txt`

# TEAM INTRUCTIONS

The data processing adapts with our current directory structure as shown below. detailed explanation can be found in **main.ipynb**.

![Screenshot](filestructure_screenshot.png)

```project_root/
└── dataset/
    ├── teammate1/
    │   ├── annotations/     ← LabelMe JSON files (one per annotated image)
    │   ├── rubbish/         ← images that contain rubbish (the ones you annotated live here)
    │   └── not_rubbish/     ← clean scenes (optional, for negatives)
    ├── teammate2/
    │   ├── annotations/
    │   ├── rubbish/
    │   └── not_rubbish/
    └── ...
```

#### **Important Rules**

- Images and Annotations should be stored locally or in Onedrive (NOT in GitHub).
- Annotation format: JSON (rectangles or polygons are OK). **I didn't include a XML file support, so let sal know if you're facing problems with annotation file type issues**
- JSON's filename should match image filename
- only put jsons for the annotated ones in 'annotations/' folder.

The code will then process the images and files and outputs them into `merged_dataset` folder.

YOLOv8nano is used for training, and evaluation metrics will be saved in `run_theme1` folder.
