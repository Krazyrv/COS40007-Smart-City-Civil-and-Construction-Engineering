import streamlit as st
import torch
from PIL import Image
import numpy as np
from pathlib import Path
import cv2
import yaml
import json
import io
import base64
import pandas as pd
import plotly.express as px
from ultralytics import YOLO

# ===============================================
# 🔧 Page Config
# ===============================================
st.set_page_config(
    page_title="Smart City Rubbish Detection",
    page_icon="🗑️",
    layout="wide"
)

# ===============================================
# 📂 Load Class Labels
# ===============================================
DATA_YAML_PATH = Path("merged_dataset/data.yaml")
with open(DATA_YAML_PATH, 'r') as f:
    data_config = yaml.safe_load(f)
    CLASS_NAMES = list(data_config['names'].values()) if isinstance(data_config['names'], dict) else data_config['names']

# ===============================================
# 🧠 Sidebar - Model Selection
# ===============================================
st.sidebar.header("Model Selection")
model_type = st.sidebar.selectbox(
    "Choose Model:",
    [
        "YOLOv8n (Object Detection)", 
        "YOLOv8s (Object Detection)",
        "YOLOv10n (Object Detection)", 
        "Faster R-CNN (Object Detection)",
        "ResNet50 (Classification)"
    ],
    key="model_choice"
)

# Model Info Card
if "YOLOv8n" in model_type:
    st.sidebar.info("""
    **YOLOv8n (Nano)**  
    - Fast, lightweight  
    - Great for CPU and small datasets  
    - Slightly less accurate than YOLOv10n  
    """)
elif "YOLOv10n" in model_type:
    st.sidebar.info("""
    **YOLOv10n (Next Gen)**  
    - Improved detection accuracy  
    - Handles small objects better  
    - Optimized for GPU inference  
    """)
elif "Faster R-CNN" in model_type:
    st.sidebar.info("""
    **Faster R-CNN**  
    - Two-stage detection  
    - Very accurate but slower  
    - Good for detailed analysis  
    """)
else:
    st.sidebar.info("""
    **ResNet50 (Classification)**  
    - Binary classification (Rubbish / Not Rubbish)  
    - Very fast  
    - Suitable for high-level detection  
    """)

# Confidence threshold
if "Classification" not in model_type:
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold:",
        min_value=0.1,
        max_value=0.9,
        value=0.15,
        step=0.05,
        help="Higher values = fewer but more confident detections",
        key="conf_slider"
    )

# Device
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.info(f"Using device: **{device}**")

# ===============================================
# ⚙️ Model Loaders
# ===============================================
@st.cache_resource
def load_yolo_model(model_path):
    return YOLO(model_path)

@st.cache_resource
def load_resnet_model(model_path, device):
    from torchvision import models
    model = models.resnet50(weights=None)
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
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 29)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

# ===============================================
# 🧩 Load Selected Model (Updated for YOLOv8s)
# ===============================================
try:
    if "YOLOv8n" in model_type:
        model = load_yolo_model("runs_models/yolov8n/weights/best.pt")
        st.sidebar.success("✅ YOLOv8n model loaded!")
    elif "YOLOv10n" in model_type:
        model = load_yolo_model("runs_models/yolov10n/weights/best.pt")
        st.sidebar.success("✅ YOLOv10n model loaded!")
    elif "Faster R-CNN" in model_type:
        model = load_frcnn_model("weights/frcnn_weights.pt", device)
        st.sidebar.success("✅ Faster R-CNN model loaded!")
    elif "YOLOv8s" in model_type:
        model = load_yolo_model("runs_models/yolo/weights/best.pt")   
        st.sidebar.success("✅ Custom YOLOv8s model loaded!")
    else:
        model = load_resnet_model("weights/resnet50_weights.pt", device)
        st.sidebar.success("✅ ResNet50 model loaded!")
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")
    st.stop()


# ===============================================
# 📸 Image Upload & Detection
# ===============================================
st.title("🗑️ Smart City Rubbish Detection")
st.markdown("Upload an image to detect rubbish and waste items using trained AI models.")

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"],
    help="Upload an image to analyze for rubbish detection"
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

                # 📊 Detection Summary
                if len(result.boxes) > 0:
                    detections = [{
                        "Class": result.names[int(box.cls[0])],
                        "Confidence": float(box.conf[0])
                    } for box in result.boxes]

                    df = pd.DataFrame(detections)
                    fig = px.bar(
                        df, x="Class", y="Confidence",
                        color="Class", text_auto=True,
                        title="Confidence per Detected Object"
                    )
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

                    st.success(f"Found **{len(df)}** object(s)")
                    st.table(df)

                    # 💾 Export Results
                    export_format = st.radio("Select export format:", ("JSON", "TXT"), horizontal=True)
                    if st.button("💾 Export Detection Results"):
                        if export_format == "JSON":
                            output_str = json.dumps(detections, indent=4)
                            mime_type, file_name = "application/json", "detections.json"
                        else:
                            output_str = "\n".join(
                                [f"{d['Class']} {d['Confidence']:.2f}" for d in detections]
                            )
                            mime_type, file_name = "text/plain", "detections.txt"

                        b64 = base64.b64encode(output_str.encode()).decode()
                        href = f'<a href="data:{mime_type};base64,{b64}" download="{file_name}">📥 Click here to download {file_name}</a>'
                        st.markdown(href, unsafe_allow_html=True)
                        st.success(f"✅ Exported {len(df)} detections as {file_name}")
                else:
                    st.info("No objects detected.")

            elif "Faster R-CNN" in model_type:
                from torchvision import transforms as T
                img_tensor = T.ToTensor()(image).to(device).unsqueeze(0)
                with torch.no_grad():
                    pred = model(img_tensor)[0]

                boxes, labels, scores = pred["boxes"].cpu(), pred["labels"].cpu(), pred["scores"].cpu()
                mask = (scores >= confidence_threshold) & (labels > 0)
                boxes, labels, scores = boxes[mask], labels[mask], scores[mask]

                img_with_boxes = np.array(image)
                for box, label, score in zip(boxes, labels, scores):
                    x1, y1, x2, y2 = box.int().tolist()
                    cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img_with_boxes, f"{CLASS_NAMES[int(label)-1]}: {score:.2f}",
                                (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                st.image(img_with_boxes, use_column_width=True)

            else:  # ResNet
                from torchvision import transforms as T
                transform = T.Compose([T.Resize((384, 384)), T.ToTensor()])
                img_tensor = transform(image).to(device).unsqueeze(0)
                with torch.no_grad():
                    logits = model(img_tensor)
                    probs = torch.softmax(logits, dim=1)[0]
                prob_not, prob_rub = probs[0].item(), probs[1].item()
                st.metric("Not Rubbish", f"{prob_not:.1%}")
                st.metric("Rubbish", f"{prob_rub:.1%}")
                st.progress(prob_not, text="Not Rubbish")
                st.progress(prob_rub, text="Rubbish")

# ===============================================
# 🎥 Live Detection (Optional)
# ===============================================
st.markdown("---")
st.header("🎥 Live Detection (Optional Demo)")
st.write("Upload a short **video clip** for detection (MP4, AVI, MOV, MPEG4).")

video_file = st.file_uploader(
    "Drag and drop or browse a video file",
    type=["mp4", "avi", "mov", "mpeg4"],
    help="Limit: 200MB per file"
)

if video_file:
    video_path = Path("temp_video.mp4")
    with open(video_path, "wb") as f:
        f.write(video_file.read())
    st.video(str(video_path))
    st.info("Running YOLO detection on the uploaded video...")

    try:
        with st.spinner("Processing video..."):
            results = model.predict(
                source=str(video_path),
                conf=confidence_threshold,
                save=True,
                project="runs/video_detect",
                name="demo",
                exist_ok=True
            )
        output_dir = Path("runs/video_detect/demo")
        processed_videos = list(output_dir.glob("*.mp4"))
        if processed_videos:
            st.success("✅ Detection complete! Displaying processed video:")
            st.video(str(processed_videos[-1]))
        else:
            st.warning("⚠️ No output video found.")
    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Upload a video to start detection.")
