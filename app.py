import streamlit as st
import torch
from PIL import Image
import numpy as np
from pathlib import Path
import cv2
import yaml
# ASSUMES frcnn, ssd, resnet MODELS ARE IN ./weights/

# Import YOLO only - other models loaded directly from torchvision
from ultralytics import YOLO

st.set_page_config(
    page_title="Smart City Rubbish Detection", page_icon="🗑️", layout="wide"
)

# Load class names from data.yaml
DATA_YAML_PATH = Path("merged_dataset/data.yaml")
with open(DATA_YAML_PATH, "r") as f:
    data_config = yaml.safe_load(f)
    CLASS_NAMES = (
        list(data_config["names"].values())
        if isinstance(data_config["names"], dict)
        else data_config["names"]
    )

st.title("🗑️ Smart City Rubbish Detection")
st.markdown(
    "Upload an image to detect rubbish and waste items using trained AI models."
)

# Sidebar for model selection
st.sidebar.header("Model Selection")
model_type = st.sidebar.selectbox(
    "Choose Model:",
    [
        "YOLOv8n (Object Detection)",
        "YOLOv10n (Object Detection)",
        "Faster R-CNN (Object Detection)",
        "ResNet50 (Classification)",
    ],
)

# Confidence threshold for object detection models
if "Classification" not in model_type:
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold:",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help="Higher values = fewer but more confident detections",
    )

# Device selection
device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
st.sidebar.info(f"Using device: **{device}**")


@st.cache_resource
def load_yolo_model(model_path):
    """Load YOLO model"""
    return YOLO(model_path)


@st.cache_resource
def load_resnet_model(model_path, device):
    """Load ResNet50 classifier"""
    from torchvision import models

    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


@st.cache_resource
def load_frcnn_model(model_path, device):
    """Load Faster R-CNN model"""
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # 29 classes = 28 rubbish types + 1 background class
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 29)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


# Load the selected model
try:
    if "YOLOv8n" in model_type:
        model_path = "runs_models/yolov8n/weights/best.pt"
        model = load_yolo_model(model_path)
        st.sidebar.success("✅ YOLOv8n model loaded!")
    elif "YOLOv10n" in model_type:
        model_path = "runs_models/yolov10n/weights/best.pt"
        model = load_yolo_model(model_path)
        st.sidebar.success("✅ YOLOv10n model loaded!")
    elif "Faster R-CNN" in model_type:
        model_path = "weights/frcnn_weights.pt"
        model = load_frcnn_model(model_path, device)
        st.sidebar.success("✅ Faster R-CNN model loaded!")
    else:
        model_path = "runs_models/resnet50_weights.pt"
        model = load_resnet_model(model_path, device)
        st.sidebar.success("✅ ResNet50 model loaded!")
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")
    st.stop()

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"],
    help="Upload an image to analyze for rubbish detection",
)

if uploaded_file is not None:
    # Display original image
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📸 Original Image")
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)

    with col2:
        st.subheader("🔍 Prediction Results")

        # Make prediction
        with st.spinner("Analyzing image..."):
            if "YOLO" in model_type:
                # YOLO object detection
                results = model(image, conf=confidence_threshold)
                result = results[0]

                # Get annotated image
                annotated_img = result.plot()
                annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                st.image(annotated_img, use_column_width=True)

                # Display detections
                if len(result.boxes) > 0:
                    st.success(f"Found **{len(result.boxes)}** object(s)")

                    # Create a dataframe of detections
                    detections = []
                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        class_name = result.names[class_id]
                        confidence = float(box.conf[0])
                        detections.append(
                            {"Class": class_name, "Confidence": f"{confidence:.2%}"}
                        )

                    st.table(detections)
                else:
                    st.info("No objects detected in the image.")

            elif "Faster R-CNN" in model_type:
                # Faster R-CNN object detection
                from torchvision import transforms as T

                # Prepare image
                img_tensor = T.ToTensor()(image).to(device).unsqueeze(0)

                # Get predictions
                with torch.no_grad():
                    pred = model(img_tensor)[0]

                # Filter predictions by confidence
                boxes = pred["boxes"].cpu()
                labels = pred["labels"].cpu()
                scores = pred["scores"].cpu()

                # Apply confidence threshold and filter out background (label 0)
                mask = (scores >= confidence_threshold) & (labels > 0)
                boxes = boxes[mask]
                labels = labels[mask]
                scores = scores[mask]

                # Draw boxes on image
                img_with_boxes = np.array(image)
                for box, label, score in zip(boxes, labels, scores):
                    x1, y1, x2, y2 = box.int().tolist()
                    # Labels are 1-28, but CLASS_NAMES is indexed 0-27
                    class_name = CLASS_NAMES[int(label) - 1]

                    # Draw rectangle
                    cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Draw label
                    label_text = f"{class_name}: {score:.2f}"
                    cv2.putText(
                        img_with_boxes,
                        label_text,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

                st.image(img_with_boxes, use_column_width=True)

                # Display detections
                if len(boxes) > 0:
                    st.success(f"Found **{len(boxes)}** object(s)")

                    detections = []
                    for label, score in zip(labels, scores):
                        detections.append(
                            {
                                "Class": CLASS_NAMES[int(label) - 1],
                                "Confidence": f"{score:.2%}",
                            }
                        )

                    st.table(detections)
                else:
                    st.info("No objects detected in the image.")

            else:
                # ResNet50 classification
                from torchvision import transforms as T

                # Prepare image
                transform = T.Compose([T.Resize((384, 384)), T.ToTensor()])
                img_tensor = transform(image).to(device).unsqueeze(0)

                # Get prediction
                with torch.no_grad():
                    logits = model(img_tensor)
                    probs = torch.softmax(logits, dim=1)[0]

                prob_not_rubbish = probs[0].item()
                prob_rubbish = probs[1].item()

                st.image(image, use_column_width=True)

                # Show prediction with color coding
                if prob_rubbish > prob_not_rubbish:
                    st.error(f"🗑️ **RUBBISH DETECTED** ({prob_rubbish:.1%} confidence)")
                else:
                    st.success(f"✨ **NO RUBBISH** ({prob_not_rubbish:.1%} confidence)")

                # Show probabilities
                st.markdown("### Prediction Probabilities")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Not Rubbish", f"{prob_not_rubbish:.1%}")
                with col_b:
                    st.metric("Rubbish", f"{prob_rubbish:.1%}")

                # Progress bars
                st.progress(prob_not_rubbish, text="Not Rubbish")
                st.progress(prob_rubbish, text="Rubbish")

else:
    st.info("👆 Upload an image to get started!")

    # Show some info about the models
    with st.expander("ℹ️ About the Models"):
        st.markdown("""
        ### Model Information
        
        **YOLOv8n & YOLOv10n (Object Detection)**
        - Fast, lightweight YOLO models
        - Detects and localizes 28 different types of rubbish items
        - Real-time detection with bounding boxes
        - Best for speed and general detection
        
        **Faster R-CNN (Object Detection)**
        - Two-stage detector with high accuracy
        - Detects 28 different types of rubbish items
        - Slower but potentially more precise than YOLO
        - Better for detailed analysis
        
        **ResNet50 (Binary Classification)**
        - Classifies entire image as containing rubbish or not
        - Simple binary prediction: "Rubbish" vs "Not Rubbish"
        - Good for quick overall assessment
        - Fastest inference time
        """)
