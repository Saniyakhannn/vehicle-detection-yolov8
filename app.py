import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import torch

# Page config
st.set_page_config(
    page_title="Vehicle Detection App",
    page_icon="🚗",
    layout="wide"
)

# Title
st.title("🚗 Vehicle Detection System")
st.markdown("### Powered by YOLOv8 | Detects: Bus, Car, Motorcycle, Truck")
st.divider()

# Load model
@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

# Sidebar
st.sidebar.title("⚙️ Settings")
confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.25,
    step=0.05
)

show_gradcam = st.sidebar.checkbox("Show Grad-CAM Heatmap", value=True)

st.sidebar.divider()
st.sidebar.markdown("### 📊 Model Info")
st.sidebar.info("""
- **Model:** YOLOv8n
- **Classes:** 4
- **mAP50:** 63.5%
- **Dataset:** 19,000 images
""")

# Upload image
uploaded_file = st.file_uploader(
    "Upload an image",
    type=['jpg', 'jpeg', 'png']
)

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(
        bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Run detection
    with st.spinner("Detecting vehicles..."):
        results = model(img, conf=confidence)
        result = results[0]

    # Count detections
    boxes = result.boxes
    total = len(boxes)

    # Class counts
    classes = ['bus', 'car', 'motorcycle', 'truck']
    class_colors = {
        'bus': '🟢',
        'car': '🔵',
        'motorcycle': '🟡',
        'truck': '🟣'
    }
    counts = {cls: 0 for cls in classes}
    for box in boxes:
        cls_id = int(box.cls[0])
        counts[classes[cls_id]] += 1

    # Show stats
    st.markdown("## 📊 Detection Results")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Vehicles", total)
    col2.metric("🟢 Bus", counts['bus'])
    col3.metric("🔵 Car", counts['car'])
    col4.metric("🟡 Motorcycle", counts['motorcycle'])
    col5.metric("🟣 Truck", counts['truck'])

    st.divider()

    # Show images
    if show_gradcam:
        col1, col2, col3 = st.columns(3)
    else:
        col1, col2 = st.columns(2)

    # Original image
    with col1:
        st.markdown("### 📷 Original Image")
        st.image(img_rgb, use_container_width=True)

    # Detection result
    detected_img = result.plot()
    detected_rgb = cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB)
    with col2:
        st.markdown("### 🎯 Detection Result")
        st.image(detected_rgb, use_container_width=True)

    # Grad-CAM
    if show_gradcam:
        with col3:
            st.markdown("### 🔥 Grad-CAM Heatmap")
            with st.spinner("Generating Grad-CAM..."):
                try:
                    pytorch_model = model.model
                    pytorch_model.eval()
                    activations = []

                    def forward_hook(module, input, output):
                        activations.append(output.detach())

                    target_layer = pytorch_model.model[9].cv2
                    handle = target_layer.register_forward_hook(
                        forward_hook)

                    img_resized = cv2.resize(img_rgb, (640, 640))
                    img_float = img_resized / 255.0
                    input_tensor = torch.from_numpy(
                        img_float).permute(2,0,1).unsqueeze(0).float()

                    with torch.no_grad():
                        pytorch_model(input_tensor)
                    handle.remove()

                    act = activations[0].squeeze(0).numpy()
                    heatmap = np.max(act, axis=0)
                    heatmap = np.maximum(heatmap, 0)
                    p_low = np.percentile(heatmap, 50)
                    p_high = np.percentile(heatmap, 99)
                    heatmap = np.clip(heatmap, p_low, p_high)
                    heatmap = (heatmap - p_low) / (
                        p_high - p_low + 1e-8)
                    heatmap = cv2.resize(heatmap, (640, 640))
                    heatmap_colored = cv2.applyColorMap(
                        np.uint8(255 * heatmap), cv2.COLORMAP_JET)
                    heatmap_rgb = cv2.cvtColor(
                        heatmap_colored, cv2.COLOR_BGR2RGB)
                    overlay = (0.4 * img_resized +
                               0.6 * heatmap_rgb).astype(np.uint8)
                    st.image(overlay, use_container_width=True)
                except Exception as e:
                    st.error(f"Grad-CAM error: {e}")

    # Detection details table
    if total > 0:
        st.divider()
        st.markdown("### 📋 Detection Details")
        data = []
        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            data.append({
                "#": i+1,
                "Class": classes[cls_id],
                "Confidence": f"{conf:.2%}",
            })
        st.table(data)
    else:
        st.warning("No vehicles detected! Try lowering the confidence threshold.")

else:
    st.info("👆 Upload an image to start detecting vehicles!")
    st.markdown("""
    ### What this app detects:
    - 🚌 **Bus**
    - 🚗 **Car**
    - 🏍️ **Motorcycle**
    - 🚚 **Truck**
    """)