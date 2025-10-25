import streamlit as st
import torch
from PIL import Image
import numpy as np
from pathlib import Path
import cv2

# Import model classes
from model.models.resnet_classifier import ResNetClassifier
from ultralytics import YOLO

st.set_page_config(
    page_title="Smart City Rubbish Detection",
    page_icon="🗑️",
    layout="wide"
)

# App Title
st.title("🗑️ Smart City Rubbish Detection")
st.markdown("Upload an image to detect rubbish and waste items using trained AI models.")

# Sidebar for model selection
st.sidebar.header("Model Selection")
model_type = st.sidebar.selectbox(
    "Choose Model:",
    ["YOLOv8n (Object Detection)", "YOLOv10n (Object Detection)", "ResNet50 (Classification)"]
)

# Confidence threshold for YOLO models
if "YOLO" in model_type:
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold:",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help="Higher values = fewer but more confident detections"
    )

# Device selection
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.info(f"Using device: **{device}**")

@st.cache_resource
def load_yolo_model(model_path):
    """Load YOLO model"""
    return YOLO(model_path)

@st.cache_resource
def load_resnet_model(model_path, device):
    """Load ResNet50 classifier"""
    # Create dummy directories for initialization
    from model.config.settings import IMAGES, LABELS, HP
    model = ResNetClassifier(IMAGES, LABELS, hp=HP, device=device)
    model.load(model_path)
    model.model.eval()
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
    help="Upload an image to analyze for rubbish detection"
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
                        detections.append({
                            "Class": class_name,
                            "Confidence": f"{confidence:.2%}"
                        })
                    
                    st.table(detections)
                else:
                    st.info("No objects detected in the image.")
                    
            else:
                # ResNet50 classification
                # Save temp file for prediction
                temp_path = Path("temp_upload.jpg")
                image.save(temp_path)
                
                predictions = model.predict([str(temp_path)])
                pred = predictions[0]
                
                # Clean up temp file
                temp_path.unlink()
                
                # Display results
                prob_not_rubbish = pred["probs"]["not_rubbish"]
                prob_rubbish = pred["probs"]["rubbish"]
                
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
        - Detects and localizes 28 different types of rubbish items
        - Categories include: bottles, cans, cardboard, furniture, litter, etc.
        - Draws bounding boxes around detected objects
        
        **ResNet50 (Binary Classification)**
        - Classifies entire image as containing rubbish or not
        - Simple binary prediction: "Rubbish" vs "Not Rubbish"
        - Good for quick overall assessment
        """)

