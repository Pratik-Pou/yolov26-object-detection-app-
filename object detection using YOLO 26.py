import streamlit as st
from ultralytics import YOLO
from PIL import Image
import io
import cv2
import tempfile
import os
import streamlit_webrtc as webrtc
import av

# Load the YOLO model

def load_model():
    return YOLO('yolo26n.pt')  # You can change to other models like 'yolo26s.pt', 'yolo26m.pt', etc.

model = load_model()

st.title("Object Detection with YOLO26")

tab1, tab2 = st.tabs(["Upload Image/Video", "Live Webcam"])

with tab1:
    st.write("Upload an image or video to detect objects.")
    
    uploaded_file = st.file_uploader("Choose an image or video...", type=["jpg", "png", "jpeg", "webp", "bmp", "tiff", "gif", "mp4", "avi", "mov", "mkv"])
    
    if uploaded_file is not None:
        if uploaded_file.type.startswith('image/'):
            # Process image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            # Perform detection
            with st.spinner('Detecting objects...'):
                results = model(image)
            
            # Display results
            st.subheader("Detection Results")
            result_image = results[0].plot()
            st.image(result_image, caption='Detected Objects', use_column_width=True)
            
            # Show detected classes
            if results[0].boxes:
                detected_classes = [model.names[int(cls)] for cls in results[0].boxes.cls]
                st.write("Detected objects:", ", ".join(set(detected_classes)))
            else:
                st.write("No objects detected.")
        
        elif uploaded_file.type.startswith('video/'):
            # Process video
            st.video(uploaded_file)
            
            # Save video to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.read())
                video_path = tmp_file.name
            
            # Process video frames
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            st.write(f"Video has {frame_count} frames at {fps} FPS. Processing every 30th frame for demo...")
            
            detections = []
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % 30 == 0:  # Process every 30th frame
                    results = model(frame)
                    if results[0].boxes:
                        detections.append((frame_idx, results[0]))
                frame_idx += 1
            
            cap.release()
            os.unlink(video_path)  # Clean up
            
            # Display detections
            st.subheader("Detection Results from Video")
            for idx, result in detections[:5]:  # Show first 5 detections
                st.write(f"Frame {idx}:")
                result_image = result.plot()
                st.image(result_image, caption=f'Detected in Frame {idx}', use_column_width=True)
                detected_classes = [model.names[int(cls)] for cls in result.boxes.cls]
                st.write("Detected objects:", ", ".join(set(detected_classes)))
            
            if not detections:
                st.write("No objects detected in the sampled frames.")
        
        else:
            st.error("Unsupported file type.")

with tab2:
    st.write("Live object detection from webcam.")
    
    class VideoProcessor(webrtc.VideoProcessorBase):
        def __init__(self):
            self.model = model
        
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            results = self.model(img)
            img_with_boxes = results[0].plot()
            return av.VideoFrame.from_ndarray(img_with_boxes, format="bgr24")
    
    webrtc.webrtc_streamer(
        key="object-detection",
        video_processor_factory=VideoProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    )

st.header("How to Train YOLO26 for Custom Detection")
st.write("""
To improve detection of specific wild animals (e.g., rare species not in COCO), you need to fine-tune the model on custom data.

### Steps:
1. **Collect Data**: Gather images of the wild animals you want to detect. Aim for 100-1000 images per class.
2. **Annotate Images**: Use tools like [LabelImg](https://github.com/heartexlabs/labelImg) or [Roboflow](https://roboflow.com/) to draw bounding boxes around the animals and label them.
3. **Prepare Dataset**: Organize into folders like `images/` and `labels/` with YOLO format annotations (.txt files).
4. **Train the Model**: Use the script below in a Python environment.

### Sample Training Script (train.py):
```python
from ultralytics import YOLO

# Load pre-trained model
model = YOLO('yolo26n.pt')

# Train on custom dataset (update paths)
model.train(data='path/to/data.yaml', epochs=50, imgsz=640)

# Save the trained model
model.save('yolo26_custom.pt')
```

5. **Update App**: Replace 'yolo26n.pt' with 'yolo26_custom.pt' in app.py.

For detailed guides, check [Ultralytics Docs](https://docs.ultralytics.com/).
""")

st.write("If you provide sample images and labels, I can help set up training!")