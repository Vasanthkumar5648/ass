import streamlit as st
import os
import time
import json
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from collections import defaultdict, deque
from datetime import datetime
from utils.detection import VideoObjectDetector, RealTimeDetection, ModelComparator
from utils.visualization import plot_class_distribution, plot_alert_timeline

# Page configuration
st.set_page_config(
    page_title="Object Detection Dashboard",
    page_icon="🔍",
    layout="wide"
)

# Sidebar navigation
st.sidebar.title("Navigation")
task = st.sidebar.radio(
    "Select Task",
    ("Video Analysis", "Real-Time Simulation", "Model Comparison")
)

# Initialize detectors (cached for performance)
@st.cache_resource
def get_video_detector():
    return VideoObjectDetector()

@st.cache_resource
def get_realtime_detector():
    return RealTimeDetection()

@st.cache_resource
def get_model_comparator():
    return ModelComparator()

# Task 1: Video Analysis
if task == "Video Analysis":
    st.title("📹 Video Object Detection Analysis")
    st.write("""
    Analyze objects in a video file. Processes every Nth frame and provides:
    - Per-frame detection results
    - Class distribution statistics
    - Frame with maximum diversity
    """)
    
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
    every_nth = st.slider("Process every N-th frame", 1, 10, 5)
    
    if uploaded_file is not None:
        # Save uploaded file
        video_path = f"temp_{uploaded_file.name}"
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if st.button("Analyze Video"):
            with st.spinner("Processing video..."):
                detector = get_video_detector()
                output_dir = "video_output"
                os.makedirs(output_dir, exist_ok=True)
                
                results = detector.detect_video(video_path, output_dir, every_nth)
                
                st.success("Processing complete!")
                
                # Show video info
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Video Information")
                    st.json({
                        "File": uploaded_file.name,
                        "Duration": f"{results['video_info']['duration_seconds']:.2f} seconds",
                        "Total Frames": results['video_info']['total_frames'],
                        "Processed Frames": results['video_info']['processed_frames']
                    })
                
                with col2:
                    st.subheader("Detection Summary")
                    st.json({
                        "Total Objects Detected": sum(results['class_counts'].values()),
                        "Unique Classes": len(results['class_counts']),
                        "Most Common Class": max(results['class_counts'], key=results['class_counts'].get)
                    })
                
                # Show class distribution
                st.subheader("Class Distribution")
                fig = plot_class_distribution(results['class_counts'])
                st.pyplot(fig)
                
                # Show sample frames
                st.subheader("Sample Detections")
                frame_files = sorted([f for f in os.listdir(output_dir) if f.startswith("frame_") and f.endswith(".jpg")])
                if frame_files:
                    sample_frames = frame_files[:5]  # Show first 5 frames
                    cols = st.columns(len(sample_frames))
                    for idx, frame_file in enumerate(sample_frames):
                        with cols[idx]:
                            st.image(Image.open(f"{output_dir}/{frame_file}"), caption=frame_file)
                
                # Download results
                st.download_button(
                    label="Download Results JSON",
                    data=json.dumps(results, indent=2),
                    file_name="detection_results.json",
                    mime="application/json"
                )

# Task 2: Real-Time Simulation
elif task == "Real-Time Simulation":
    st.title("🎥 Real-Time Stream Simulation")
    st.write("""
    Simulate a real-time video stream with object detection.
    Alerts are triggered when 3+ people appear in 5 consecutive frames.
    """)
    
    sim_option = st.radio(
        "Simulation Source",
        ("Upload Video", "Use Webcam")
    )
    
    if sim_option == "Upload Video":
        sim_file = st.file_uploader("Upload simulation video", type=["mp4", "avi", "mov"])
    else:
        st.warning("Webcam support requires browser permission and may not work on all servers.")
    
    every_nth = st.slider("Process every N-th frame", 1, 10, 3)
    alert_threshold = st.slider("Alert threshold (people)", 1, 10, 3)
    buffer_size = st.slider("Consecutive frames for alert", 1, 10, 5)
    
    if st.button("Start Simulation"):
        if sim_option == "Upload Video" and sim_file is not None:
            # Save uploaded file
            video_path = f"temp_{sim_file.name}"
            with open(video_path, "wb") as f:
                f.write(sim_file.getbuffer())
            
            detector = get_realtime_detector()
            detector.person_buffer = deque(maxlen=buffer_size)
            
            output_path = "simulation_output.mp4"
            
            # Create placeholders
            video_placeholder = st.empty()
            alert_placeholder = st.empty()
            stats_placeholder = st.empty()
            
            # Process video
            cap = cv2.VideoCapture(video_path)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            frame_count = 0
            alert_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                processed_frame = frame.copy()
                
                if frame_count % every_nth == 0:
                    detections = detector.detect_frame(frame)
                    person_count = sum(1 for d in detections if d['class'] == 'person')
                    detector.person_buffer.append(person_count)
                    
                    # Check for alert condition
                    if len(detector.person_buffer) == buffer_size and all(count >= alert_threshold for count in detector.person_buffer):
                        alert_msg = f"🚨 Crowd Detected! {person_count} people in frame {frame_count}"
                        alert_placeholder.warning(alert_msg)
                        alert_count += 1
                        
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        detector.alert_log.append({
                            'timestamp': timestamp,
                            'frame': frame_count,
                            'message': alert_msg,
                            'person_count': person_count
                        })
                        
                        # Draw alert on frame
                        cv2.putText(processed_frame, "ALERT: Crowd Detected!", (50, 50), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    # Draw detections
                    processed_frame = detector.draw_boxes(processed_frame, detections)
                
                # Display frame
                video_placeholder.image(processed_frame, channels="BGR", use_column_width=True)
                
                # Update stats
                stats_placeholder.info(f"""
                Frame: {frame_count} | People in frame: {person_count if frame_count % every_nth == 0 else 'N/A'}
                Alerts triggered: {alert_count}
                Buffer: {list(detector.person_buffer)}
                """)
                
                frame_count += 1
                time.sleep(1/fps)  # Simulate real-time
            
            cap.release()
            
            # Show final results
            st.success("Simulation complete!")
            st.subheader("Alert Log")
            st.json(detector.alert_log)
            
            # Plot timeline
            fig = plot_alert_timeline(detector.alert_history)
            st.pyplot(fig)
            
            # Download log
            st.download_button(
                label="Download Alert Log",
                data=json.dumps(detector.alert_log, indent=2),
                file_name="alert_log.json",
                mime="application/json"
            )

# Task 3: Model Comparison
elif task == "Model Comparison":
    st.title("⚖️ Model Comparison")
    st.write("""
    Compare two object detection models on the same set of images.
    Metrics include inference time, detection count, and class diversity.
    """)
    
    model_options = {
        "YOLOv5 Nano": "yolov5n.pt",
        "YOLOv5 Small": "yolov5s.pt",
        "YOLOv5 Medium": "yolov5m.pt"
    }
    
    col1, col2 = st.columns(2)
    with col1:
        model1 = st.selectbox(
            "Select Model 1",
            list(model_options.keys()),
            index=0
        )
    with col2:
        model2 = st.selectbox(
            "Select Model 2",
            list(model_options.keys()),
            index=1
        )
    
    uploaded_images = st.file_uploader(
        "Upload test images (max 10)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )
    
    if uploaded_images and st.button("Run Comparison"):
        # Limit to 10 images
        test_images = uploaded_images[:10]
        
        with st.spinner("Running comparison..."):
            # Save images to temp directory
            input_dir = "comparison_input"
            os.makedirs(input_dir, exist_ok=True)
            for img in test_images:
                with open(f"{input_dir}/{img.name}", "wb") as f:
                    f.write(img.getbuffer())
            
            # Run comparison
            comparator = ModelComparator(
                model1_path=model_options[model1],
                model2_path=model_options[model2]
            )
            
            output_dir = "comparison_output"
            results = comparator.compare_models(input_dir, output_dir)
            
            st.success("Comparison complete!")
            
            # Show summary
            st.subheader("Comparison Summary")
            summary_df = pd.DataFrame({
                "Metric": ["Average Inference Time (s)", "Average Detection Count", "Average Class Diversity"],
                model1: [
                    results['average_inference_time']['model1'],
                    results['average_detection_count']['model1'],
                    results['average_class_diversity']['model1']
                ],
                model2: [
                    results['average_inference_time']['model2'],
                    results['average_detection_count']['model2'],
                    results['average_class_diversity']['model2']
                ],
                "Difference": [
                    results['average_inference_time']['difference'],
                    results['average_detection_count']['difference'],
                    results['average_class_diversity']['difference']
                ]
            })
            st.dataframe(summary_df.style.highlight_max(axis=1))
            
            # Show sample comparisons
            st.subheader("Sample Comparisons")
            output_files = sorted(os.listdir(output_dir))
            model1_files = [f for f in output_files if f.endswith("_model1.jpg")]
            
            for sample in model1_files[:3]:  # Show first 3 comparisons
                base_name = sample.replace("_model1.jpg", "")
                col1, col2 = st.columns(2)
                with col1:
                    st.image(Image.open(f"{output_dir}/{base_name}_model1.jpg"), caption=f"{model1}")
                with col2:
                    st.image(Image.open(f"{output_dir}/{base_name}_model2.jpg"), caption=f"{model2}")
            
            # Download results
            st.download_button(
                label="Download Full Results",
                data=pd.read_csv(f"{output_dir}/per_image_results.csv").to_csv(index=False),
                file_name="model_comparison.csv",
                mime="text/csv"
            )
