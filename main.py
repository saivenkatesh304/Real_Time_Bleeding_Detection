import streamlit as st
import time
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import segmentation_models_pytorch as smp
import tempfile
import os
import pandas as pd
from pygame import mixer  # For playing alert sounds

# Initialize pygame mixer for sound alerts
mixer.init()
try:
    alert_sound = mixer.Sound("sound.mp3")  # Load your alert sound file
except:
    st.warning("Alert sound file not found. Continuing without sound alerts.")

# Load model once
@st.cache_resource(show_spinner=False)
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.to(device)
    model.eval()
    return model, device

model, device = load_model()

# Common transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Enhanced Home page
def home_page():
    st.title("Real-Time Bleeding Detection in Surgery Using Deep Learning-Based Semantic Segmentation")
    st.markdown("""
    <style>
    .metric-card {
        padding: 15px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 10px 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .header-text {
        color: #2c3e50;
    }
    .plot-container {
        display: flex;
        flex-wrap: wrap;
        gap: 20px;
        justify-content: center;
    }
    .plot-item {
        flex: 1 1 300px;
        max-width: 400px;
    }
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
        gap: 15px;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Project description
    st.markdown("""
    ## 🚀 Project Overview
    This Robotic Surgery Alert System uses deep learning to identify and highlight potential areas of concern 
    during robotic-assisted surgeries. The model was trained to segment critical regions in real-time, 
    providing visual feedback to surgical teams.
    """)
    
    st.markdown("---")
    
    # Training metrics section
    st.markdown("## 📊 Model Training Metrics")
    
    # Display CSV data
    st.markdown("### Training Logs")
    try:
        metrics_df = pd.read_csv("training_metrics.csv")
        st.dataframe(metrics_df.style.background_gradient(cmap='Blues'), use_container_width=True)
    except:
        st.warning("Training metrics file not found")
    
    st.markdown("---")
    
    # Display training plots in a grid layout
    st.markdown("### Training Progress")
    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
    
    plot_files = [
        "train_loss.png",
        "val_accuracy.png",
        "val_f1.png",
        "val_iou.png",
        "val_precision.png",
        "val_recall.png",
        "val_specificity.png"
    ]
    
    for plot_file in plot_files:
        try:
            st.markdown(f'<div class="plot-item">', unsafe_allow_html=True)
            st.image(plot_file, caption=plot_file.replace('.png', '').replace('_', ' ').title())
            st.markdown('</div>', unsafe_allow_html=True)
        except:
            st.warning(f"Plot {plot_file} not found")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Key metrics summary
    st.markdown("---")
    st.markdown("## 🏆 Key Performance Metrics")
    
    # Create a grid layout for all metrics
    st.markdown('<div class="metrics-grid">', unsafe_allow_html=True)
    
    # List of all metrics with their best values (replace with your actual best values)
    metrics = [
    {"label": "Best Validation IoU", "value": "0.703"},
    {"label": "Best F1 Score", "value": "0.82"},
    {"label": "Best Accuracy", "value": "0.92"},
    {"label": "Best Precision", "value": "0.78"},
    {"label": "Best Recall", "value": "0.98"},
    {"label": "Best Specificity", "value": "0.94"},
    {"label": "Training Epochs", "value": "50"},
    {"label": "Final Loss", "value": "0.076"}
]

    for metric in metrics:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(label=metric["label"], value=metric["value"])
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <p>Developed for Robotic Surgical Assistance</p>
    </div>
    """, unsafe_allow_html=True)


# Single image test with original, mask, and output
def single_image_test():
    st.header("Single Image Test")
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        col1, col2, col3 = st.columns(3)
        
        # Original Image
        image = Image.open(uploaded_file).convert("RGB")
        with col1:
            st.image(image, caption="Original Image", use_container_width=True)

        # Process image
        input_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            prob_map = torch.sigmoid(output).squeeze().cpu().numpy()

        # Predicted Mask
        mask = (prob_map > 0.5).astype(np.uint8)
        mask_img = Image.fromarray((mask * 255).astype(np.uint8)).resize(image.size)
        with col2:
            st.image(mask_img, caption="Predicted Mask", use_container_width=True)

        # Output with contours
        frame = np.array(image)
        mask_resized = np.array(mask_img)
        contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = 500
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        cv2.drawContours(frame, filtered_contours, -1, (255, 0, 0), 2)
        with col3:
            st.image(frame, caption="Output with Outlines", use_container_width=True)

# Video test with direct live output
def video_test():
    st.header("Video Test")
    uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
    
    # Add a threshold slider for sensitivity
    threshold = st.slider("Detection Sensitivity Threshold", 0.1, 0.9, 0.5, 0.05)
    
    if uploaded_video:
        # Save the video to a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        video_path = tfile.name
        tfile.close()  # Close the file handle to avoid permission issues
        
        # Display the original video
        st.video(video_path)
        
        # Process and display frame by frame
        st.write("Live Processing Output:")
        video_placeholder = st.empty()
        
        # Alert state tracking
        alert_active = False
        last_alert_time = 0
        alert_cooldown = 1.0  # seconds between alerts
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_delay = 1.0 / fps if fps > 0 else 0.03  # Default to ~30fps if fps is invalid
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                input_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                input_tensor = transform(input_image).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    output = model(input_tensor)
                    prob_map = torch.sigmoid(output).squeeze().cpu().numpy()
                
                mask = (prob_map > threshold).astype(np.uint8)
                mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                min_area = 500
                filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
                
                # Check if bleeding is detected (contours present)
                if len(filtered_contours) > 0:
                    current_time = time.time()
                    if current_time - last_alert_time > alert_cooldown:
                        try:
                            alert_sound.play()
                            last_alert_time = current_time
                            alert_active = True
                        except:
                            pass
                else:
                    alert_active = False
                
                # Draw contours and alert status
                cv2.drawContours(frame, filtered_contours, -1, (255, 0, 0), 2)
                
                # Add alert status text to the frame
                alert_text = "ALERT: BLEEDING DETECTED!" if alert_active else "Status: Normal"
                alert_color = (0, 0, 255) if alert_active else (0, 255, 0)
                cv2.putText(frame, alert_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, alert_color, 2, cv2.LINE_AA)
                
                # Display processed frame
                video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                
                # Control playback speed (add small delay to match original FPS)
                time.sleep(frame_delay)
                
        finally:
            cap.release()
            try:
                os.remove(video_path)
            except PermissionError:
                st.warning("Could not delete temporary video file (still in use). It will be removed automatically later.")

# Real-time camera test
def real_time_camera_test():
    st.header("Real-Time Camera Test")
    run = st.checkbox('Run Camera')
    FRAME_WINDOW = st.image([])

    if run:
        cap = cv2.VideoCapture(0)
        while run:
            ret, frame = cap.read()
            if not ret:
                st.write("Failed to capture camera frame.")
                break

            input_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            input_tensor = transform(input_image).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_tensor)
                prob_map = torch.sigmoid(output).squeeze().cpu().numpy()

            mask = (prob_map > 0.5).astype(np.uint8)
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            min_area = 500
            filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
            cv2.drawContours(frame, filtered_contours, -1, (255, 0, 0), 2)

            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)

            if not st.session_state.get('run_camera', True):
                break
        cap.release()

# Streamlit app navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Single Image Test", "Video Test", "Real-Time Camera Test"])

if page == "Home":
    home_page()
elif page == "Single Image Test":
    single_image_test()
elif page == "Video Test":
    video_test()
elif page == "Real-Time Camera Test":
    real_time_camera_test()