import streamlit as st
import cv2
import torch
import torch.nn as nn
import numpy as np
import timm
import os
from PIL import Image
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import time

# ==========================================
# 1. CONFIGURATION & MODEL LOADING
# ==========================================
st.set_page_config(page_title="Glaucoma Detection AI", layout="wide")

@st.cache_resource
def load_swin_model(model_path):
    # Model Checking
    if not os.path.exists(model_path):
        st.error(f" Model file `{model_path}` not found!")
        return None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=1)
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device

# Model Path
MODEL_PATH = "swin_glaucoma_model_best.pth" 

# ==========================================
# 2. PREPROCESSING & XAI FUNCTIONS
# ==========================================
def run_preprocessing(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    contrast = gray.std()
    if brightness < 15 or contrast < 20:
        return None, "Poor Image Quality (Too Dark/Blurry)"

    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None, "Failed to Detect Eyeball"
    
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    side = max(w, h)
    cx, cy = x + w//2, y + h//2
    x1, y1 = max(0, cx - side//2), max(0, cy - side//2)
    img_cropped = img_bgr[y1:y1+side, x1:x1+side]

    g_channel = img_cropped[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    g_enhanced = clahe.apply(g_channel)
    
    merged = cv2.merge([g_enhanced, g_enhanced, g_enhanced])
    final_224 = cv2.resize(merged, (224, 224), interpolation=cv2.INTER_AREA)
    
    return final_224, "Passed"

def reshape_transform(tensor, height=7, width=7):
    # Specific for Swin Transformer architecture
    result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result

# ==========================================
# 3. USER INTERFACE (UI) & STYLING
# ==========================================
st.markdown("""
    <style>
    [data-testid="stFileUploaderDropzone"] {
        padding: 4rem 1rem;
        border-radius: 20px;
        border: 2px dashed #4CAF50;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: #2e7d32;
        background-color: rgba(76, 175, 80, 0.1); 
    }
    .stMetric {
        padding: 10px;
        border-radius: 10px;
        border: 1px solid rgba(128, 128, 128, 0.2);
    }
    </style>
    """, unsafe_allow_html=True)

st.title("AI-Based Early Glaucoma Detection System")
st.write("Swin Transformer - Team BahteraMelaju")

uploaded_file = st.file_uploader("Upload or Drag & Drop Eye Fundus Photo Here", type=["jpg", "jpeg", "png"])

if uploaded_file:
    model, device = load_swin_model(MODEL_PATH)
    
    if model is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_original = cv2.imdecode(file_bytes, 1)
        processed_img, status = run_preprocessing(img_original)

        if processed_img is None:
            st.error(f" {status}")
        else:
            start_time = time.time()
            with st.spinner(' AI is analyzing the optical nerve structure...'):
                input_tensor = processed_img.astype(np.float32) / 255.0
                input_tensor = (input_tensor - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
                input_tensor = torch.from_numpy(input_tensor).permute(2, 0, 1).unsqueeze(0).to(device).float()

                with torch.no_grad():
                    output = model(input_tensor)
                    prob = torch.sigmoid(output).item()

                    if prob > 0.5:
                        prediction = "GLAUCOMA (GON+)"
                        confidence = prob
                        color = "red"
                    else:
                        prediction = "NORMAL (GON-)"
                        confidence = 1 - prob
                        color = "green"

                target_layers = [model.layers[-1].blocks[-1].norm2]
                cam = GradCAMPlusPlus(model=model, target_layers=target_layers, reshape_transform=reshape_transform)
                grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(0)])[0, :]
                img_float = processed_img.astype(np.float32) / 255.0
                cam_image = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)

            duration = time.time() - start_time
            
            st.divider()
            col1, col2, col3 = st.columns(3)
            with col1:
                st.subheader("1. Original Image")
                st.image(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB), use_column_width=True)
            with col2:
                st.subheader("2. Preprocessing Result")
                st.image(processed_img, use_column_width=True)
            with col3:
                st.subheader("3. AI Focus Analysis")
                st.image(cam_image, use_column_width=True)

            st.markdown(f"### Prediction Result: :{color}[{prediction}]")
            m1, m2 = st.columns(2)
            m1.metric("Confidence Score", f"{confidence*100:.2f}%")
            m2.metric("Analysis Speed", f"{duration:.2f} seconds")

            if prob > 0.5:
                st.warning(" **Warning:** The model detected signs of glaucoma. We strongly recommend further examination by an ophthalmologist.")
            else:
                st.success(" **Result:** The optic nerve condition appears normal within the scope of the AI model analysis.")

st.divider()
st.caption("AI-Based Glaucoma Screening App | BahteraMelaju | This tool is not a final medical diagnosis.")
