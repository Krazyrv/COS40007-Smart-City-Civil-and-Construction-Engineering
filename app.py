import streamlit as st
import torch
from PIL import Image
import numpy as np
from pathlib import Path
import cv2
import yaml
import json
import base64
import pandas as pd
import plotly.express as px
from ultralytics import YOLO


st.set_page_config(
    page_title="Smart City Rubbish Detection", page_icon="🗑️", layout="wide"
)


DATA_YAML_PATH = Path("merged_dataset/data.yaml")
with open(DATA_YAML_PATH, "r") as f:
    data_config = yaml.safe_load(f)
    CLASS_NAMES = (
        list(data_config["names"].values())
        if isinstance(data_config["names"], dict)
        else data_config["names"]
    )


def get_result_name(names, cls_id: int) -> str:
    """Resolve the class label returned by Ultralytics results."""
    if isinstance(names, dict):
        return names.get(cls_id, str(cls_id))
    if isinstance(names, (list, tuple)) and 0 <= cls_id < len(names):
        return names[cls_id]
    return str(cls_id)


st.title("🗑️ Smart City Rubbish Detection")
st.markdown(
    "Upload an image to detect rubbish and waste items using trained AI models."
)

st.sidebar.header("Model Selection")
model_type = st.sidebar.selectbox(
    "Choose Model:",
    [
        "YOLOv8n (Object Detection)",
        "YOLOv8s (Object Detection)",
        "YOLOv10n (Object Detection)",
        "Faster R-CNN (Object Detection)",
        "ResNet50 (Classification)",
    ],
    key="model_choice",
)

if "YOLOv8n" in model_type:
    st.sidebar.info(
        """
    **YOLOv8n (Nano)**  
    - Fast, lightweight  
    - Great for CPU and small datasets  
    - Slightly less accurate than YOLOv10n  
    """
    )
elif "YOLOv10n" in model_type:
    st.sidebar.info(
        """
    **YOLOv10n (Next Gen)**  
    - Improved detection accuracy  
    - Handles small objects better  
    - Optimized for GPU inference  
    """
    )
elif "Faster R-CNN" in model_type:
    st.sidebar.info(
        """
    **Faster R-CNN**  
    - Two-stage detection  
    - Very accurate but slower  
    - Good for detailed analysis  
    """
    )
else:
    st.sidebar.info(
        """
    **ResNet50 (Classification)**  
    - Binary classification (Rubbish / Not Rubbish)  
    - Very fast  
    - Suitable for high-level detection  
    """
    )


confidence_threshold = 0.25
if "Classification" not in model_type:
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold:",
        min_value=0.1,
        max_value=0.9,
        value=0.15,
        step=0.05,
        help="Higher values = fewer but more confident detections",
        key="conf_slider",
    )


device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
st.sidebar.info(f"Using device: **{device}**")


def resolve_weight_path(*candidates: str) -> str:
    """Return the first existing weight path from candidates, fallback to last."""
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return str(path)
    return str(candidates[-1])


@st.cache_resource
def load_yolo_model(model_path):
    return YOLO(model_path)


@st.cache_resource
def load_resnet_model(model_path, device):
    from torchvision import models

    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


@st.cache_resource
def load_frcnn_model(model_path, device):
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(CLASS_NAMES) + 1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


try:
    if "YOLOv8n" in model_type:
        weight_path = resolve_weight_path(
            "runs_models/yolov8n/weights/best.pt",
            "runs_models/yolov8n2/weights/best.pt",
            "yolov8n.pt",
        )
        model = load_yolo_model(weight_path)
        st.sidebar.success("✅ YOLOv8n model loaded!")
    elif "YOLOv8s" in model_type:
        weight_path = resolve_weight_path(
            "runs_models/yolov8s/weights/best.pt",
            "yolov8s.pt",
        )
        model = load_yolo_model(weight_path)
        st.sidebar.success("✅ YOLOv8s model loaded!")
    elif "YOLOv10n" in model_type:
        weight_path = resolve_weight_path(
            "runs_models/yolov10n6/weights/best.pt",
            "runs_models/yolov10n/weights/best.pt",
            "yolov10n.pt",
        )
        model = load_yolo_model(weight_path)
        st.sidebar.success("✅ YOLOv10n model loaded!")
    elif "Faster R-CNN" in model_type:
        model = load_frcnn_model("weights/frcnn_weights.pt", device)
        st.sidebar.success("✅ Faster R-CNN model loaded!")
    else:
        weight_path = resolve_weight_path("runs_models/resnet50_weights.pt")
        model = load_resnet_model(weight_path, device)
        st.sidebar.success("✅ ResNet50 model loaded!")
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")
    st.stop()


uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"],
    help="Upload an image to analyze for rubbish detection",
    key="image_uploader",
)

if uploaded_file:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Original Image")
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)

    with col2:
        st.subheader("🔍 Prediction Results")

        with st.spinner("Analyzing image..."):
            if "YOLO" in model_type:
                results = model(image, conf=confidence_threshold)
                result = results[0]

                annotated_img = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)
                st.image(annotated_img, use_column_width=True)

                if len(result.boxes) > 0:
                    detections = []
                    for box in result.boxes:
                        cls_id = int(box.cls[0])
                        cls_name = get_result_name(result.names, cls_id)
                        conf = float(box.conf[0])
                        detections.append({"Class": cls_name, "Confidence": conf})

                    df = pd.DataFrame(detections)
                    fig = px.bar(
                        df,
                        x="Class",
                        y="Confidence",
                        color="Class",
                        text_auto=".2f",
                        title="Confidence per Detected Object",
                    )
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

                    st.success(f"Found **{len(df)}** object(s)")
                    df_display = df.copy()
                    df_display["Confidence"] = df_display["Confidence"].map(
                        lambda v: f"{v:.2%}"
                    )
                    st.table(df_display)

                    export_format = st.radio(
                        "Select export format:", ("JSON", "TXT"), horizontal=True
                    )
                    if st.button("💾 Export Detection Results"):
                        if export_format == "JSON":
                            output_str = json.dumps(detections, indent=4)
                            mime_type, file_name = "application/json", "detections.json"
                        else:
                            output_str = "\n".join(
                                [
                                    f"{d['Class']} {d['Confidence']:.2f}"
                                    for d in detections
                                ]
                            )
                            mime_type, file_name = "text/plain", "detections.txt"

                        b64 = base64.b64encode(output_str.encode()).decode()
                        href = (
                            f'<a href="data:{mime_type};base64,{b64}" '
                            f'download="{file_name}">📥 Click here to download {file_name}</a>'
                        )
                        st.markdown(href, unsafe_allow_html=True)
                        st.success(f"✅ Exported {len(df)} detections as {file_name}")
                else:
                    st.info("No objects detected in the image.")

            elif "Faster R-CNN" in model_type:
                from torchvision import transforms as T

                img_tensor = T.ToTensor()(image).to(device).unsqueeze(0)
                with torch.no_grad():
                    pred = model(img_tensor)[0]

                boxes = pred["boxes"].cpu()
                labels = pred["labels"].cpu()
                scores = pred["scores"].cpu()

                mask = (scores >= confidence_threshold) & (labels > 0)
                boxes = boxes[mask]
                labels = labels[mask]
                scores = scores[mask]

                img_with_boxes = np.array(image)
                for box, label, score in zip(boxes, labels, scores):
                    x1, y1, x2, y2 = box.int().tolist()
                    label_text = f"{CLASS_NAMES[int(label) - 1]}: {score:.2f}"
                    cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
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

                if len(boxes) > 0:
                    st.success(f"Found **{len(boxes)}** object(s)")
                    det_rows = [
                        {
                            "Class": CLASS_NAMES[int(label) - 1],
                            "Confidence": f"{score:.2%}",
                        }
                        for label, score in zip(labels, scores)
                    ]
                    st.table(det_rows)
                else:
                    st.info("No objects detected in the image.")

            else:  # ResNet classification
                from torchvision import transforms as T

                transform = T.Compose([T.Resize((384, 384)), T.ToTensor()])
                img_tensor = transform(image).to(device).unsqueeze(0)
                with torch.no_grad():
                    logits = model(img_tensor)
                    probs = torch.softmax(logits, dim=1)[0]

                prob_not_rubbish = probs[0].item()
                prob_rubbish = probs[1].item()

                st.image(image, use_column_width=True)

                if prob_rubbish > prob_not_rubbish:
                    st.error(
                        f"🗑️ **RUBBISH DETECTED** ({prob_rubbish:.1%} confidence)"
                    )
                else:
                    st.success(
                        f"✨ **NO RUBBISH** ({prob_not_rubbish:.1%} confidence)"
                    )

                st.markdown("### Prediction Probabilities")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Not Rubbish", f"{prob_not_rubbish:.1%}")
                with col_b:
                    st.metric("Rubbish", f"{prob_rubbish:.1%}")

                st.progress(prob_not_rubbish, text="Not Rubbish")
                st.progress(prob_rubbish, text="Rubbish")
else:
    st.info("👆 Upload an image to get started!")

    with st.expander("ℹ️ About the Models"):
        st.markdown(
            """
        ### Model Information

        **YOLOv8n & YOLOv10n (Object Detection)**
        - Fast, lightweight YOLO models
        - Detect and localize multiple rubbish categories
        - Real-time detection with bounding boxes
        - Best for speed and general detection

        **YOLOv8s (Object Detection)**
        - Larger backbone for improved accuracy
        - Still suitable for mid-range devices
        - Stronger at picking small or cluttered items

        **Faster R-CNN (Object Detection)**
        - Two-stage detector with high accuracy
        - Slower but more precise on complex scenes
        - Useful for detailed offline analysis

        **ResNet50 (Binary Classification)**
        - Classifies the entire image as rubbish or not
        - Simple binary prediction
        - Good for quick overall assessment
        - Fastest inference time
        """
        )


st.markdown("---")
st.header("🎥 Live Detection (Optional Demo)")

if "YOLO" not in model_type:
    st.info("Switch to a YOLO model to enable video detection.")
else:
    st.write("Upload a short **video clip** for detection (MP4, AVI, MOV, MPEG4).")

    video_file = st.file_uploader(
        "Drag and drop or browse a video file",
        type=["mp4", "avi", "mov", "mpeg4"],
        help="Limit: 200MB per file",
        key="video_uploader",
    )

    if video_file:
        video_path = Path("temp_video.mp4")
        with open(video_path, "wb") as f:
            f.write(video_file.read())
        st.video(str(video_path))
        st.info("Running YOLO detection on the uploaded video...")

        try:
            with st.spinner("Processing video..."):
                model.predict(
                    source=str(video_path),
                    conf=confidence_threshold,
                    save=True,
                    project="runs/video_detect",
                    name="demo",
                    exist_ok=True,
                )
            output_dir = Path("runs/video_detect/demo")
            processed_videos = sorted(output_dir.glob("*.mp4"))
            if processed_videos:
                st.success("✅ Detection complete! Displaying processed video:")
                st.video(str(processed_videos[-1]))
            else:
                st.warning("⚠️ No output video found.")
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.info("Upload a video to start detection.")

