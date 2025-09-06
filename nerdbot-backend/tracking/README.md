# Object Tracking Module

Advanced computer vision system for real-time object detection, tracking, and pose estimation using Hailo AI acceleration.

## Overview

The tracking module provides high-performance computer vision capabilities including object detection, multi-object tracking, pose estimation, and instance segmentation. It leverages Hailo AI hardware acceleration for real-time inference on Raspberry Pi 5.

## Features

- **Real-time Object Detection**: YOLOv8 with Hailo acceleration
- **Multi-Object Tracking**: Track multiple objects across frames
- **Pose Estimation**: Human pose detection and keypoint tracking
- **Instance Segmentation**: Pixel-level object segmentation
- **Optimized Performance**: Hardware-accelerated inference at 20-30 FPS

## AI Models

### Detection Models
- **yolov8n.hef**: Nano model for fast detection
- **yolov8s.hef**: Small model for balanced performance
- **yolov8m.hef**: Medium model for higher accuracy

### Pose Models  
- **yolov8n-pose.hef**: Human pose estimation
- **yolov8s-pose.hef**: Enhanced pose detection

### Segmentation Models
- **yolov8n-seg.hef**: Instance segmentation
- **yolov8s-seg.hef**: Higher quality segmentation

## Usage

### Object Detection

```python
from basic_pipelines.detection import DetectionPipeline

# Initialize pipeline
pipeline = DetectionPipeline("yolov8n.hef")

# Run detection on image
results = pipeline.run("image.jpg")

# Process results
for detection in results:
    label = detection['label']
    confidence = detection['confidence'] 
    bbox = detection['bbox']  # [x, y, w, h]
    print(f"Found {label} with {confidence:.2f} confidence")
```

### Pose Estimation

```python
from basic_pipelines.pose_estimation import PoseEstimationPipeline

# Initialize pose pipeline
pose_pipeline = PoseEstimationPipeline("yolov8n-pose.hef")

# Detect poses
poses = pose_pipeline.run("image.jpg")

# Process keypoints
for pose in poses:
    keypoints = pose['keypoints']  # 17 body keypoints
    confidence = pose['confidence']
    # keypoints format: [[x, y, confidence], ...]
```

### Instance Segmentation

```python
from basic_pipelines.instance_segmentation import InstanceSegmentationPipeline

# Initialize segmentation
seg_pipeline = InstanceSegmentationPipeline("yolov8n-seg.hef")

# Run segmentation  
results = seg_pipeline.run("image.jpg")

# Process masks
for result in results:
    mask = result['mask']  # Binary mask
    label = result['label']
    bbox = result['bbox']
```

### Optimized Tracker

```python
from tracking.optimized_tracker import OptimizedTracker

# Initialize tracker
tracker = OptimizedTracker()

# Process video stream
for frame in video_stream:
    detections = tracker.track_frame(frame)
    
    for detection in detections:
        track_id = detection['track_id']
        label = detection['label']
        bbox = detection['bbox']
        # Draw tracking results
```

## Configuration

### Model Configuration

Models are stored in `resources/` directory:
```
resources/
├── yolov8n.hef              # Detection model
├── yolov8n-pose.hef         # Pose estimation
├── yolov8n-seg.hef          # Segmentation
└── detection_config.json    # Detection parameters
```

### Pipeline Settings

```python
# Detection parameters
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
MAX_DETECTIONS = 100

# Tracking parameters  
MAX_AGE = 30           # Frames to keep lost tracks
MIN_HITS = 3           # Hits required to start track
IOU_THRESHOLD = 0.3    # Intersection over Union threshold
```

## Hailo Integration

### Hardware Setup
- **Hailo AI Accelerator**: Hailo-8 or Hailo-8L
- **Connection**: USB 3.0 or PCIe
- **Power**: Sufficient USB power or external power
- **Drivers**: HailoRT and TAPPAS installed

### Model Compilation
Models are pre-compiled for Hailo hardware:
```bash
# Download pre-compiled models
./download_resources.sh --all

# Or compile custom models
hailo parser yolov8n.onnx --har yolov8n.har
hailo compiler yolov8n.har --hef yolov8n.hef
```

## Performance Metrics

### Detection Performance
- **YOLOv8n**: 25-30 FPS at 640x480
- **YOLOv8s**: 20-25 FPS at 640x480  
- **YOLOv8m**: 15-20 FPS at 640x480

### Accuracy (COCO dataset)
- **YOLOv8n**: 37.3% mAP@0.5:0.95
- **YOLOv8s**: 44.9% mAP@0.5:0.95
- **YOLOv8m**: 50.2% mAP@0.5:0.95

### Resource Usage
- **Memory**: 1-2GB depending on model
- **CPU**: 20-40% (preprocessing/postprocessing)
- **Hailo**: 80-100% utilization during inference

## Object Classes

The models detect 80 COCO classes including:
- **People**: person
- **Vehicles**: car, truck, bus, motorcycle, bicycle
- **Animals**: cat, dog, bird, horse, cow
- **Objects**: chair, table, laptop, phone, cup
- **Sports**: ball, frisbee, skateboard, skis

## Flask Integration

The tracking module integrates with Flask server:

```python
@app.route('/api/detection', methods=['POST'])
def detect_objects():
    # Get image from request
    image = request.files['image']
    
    # Run detection
    results = detection_pipeline.run(image)
    
    # Return JSON results
    return jsonify({
        'detections': results,
        'count': len(results),
        'inference_time': pipeline.last_inference_time
    })
```

## Real-time Streaming

Integration with video streaming:

```python
def generate_frames_with_detection():
    while True:
        # Capture frame
        frame = camera.capture_array()
        
        # Run detection
        detections = detection_pipeline.run(frame)
        
        # Draw bounding boxes
        annotated_frame = draw_detections(frame, detections)
        
        # Encode and stream
        yield encode_frame(annotated_frame)
```

## Requirements

- **HailoRT**: Hailo runtime library
- **TAPPAS**: Hailo TAPPAS framework  
- **OpenCV**: Image processing
- **NumPy**: Numerical computations
- **Pillow**: Image handling
- **Hailo Hardware**: Hailo-8 or Hailo-8L accelerator

## Installation

```bash
# Install Hailo dependencies
sudo apt install hailo-all

# Install Python packages
cd /home/mark/nerdbot-backend
source setup_env.sh
pip install -r requirements.txt

# Download AI models
./download_resources.sh --all
```

## Troubleshooting

### Hailo Hardware Issues
1. Check hardware connection: `lsusb | grep Hailo`
2. Verify drivers: `hailortcli scan`
3. Test basic inference: `hailortcli run yolov8n.hef`
4. Check power supply to Hailo device

### Model Loading Errors
1. Verify model files exist: `ls resources/*.hef`
2. Check model compatibility with HailoRT version
3. Validate model file integrity
4. Try different model (yolov8n vs yolov8s)

### Performance Issues
1. Check Hailo utilization: `hailortcli monitor`
2. Verify input resolution settings
3. Reduce batch size or model complexity
4. Check CPU/memory usage: `htop`

### Inference Errors
1. Validate input image format (RGB, correct size)
2. Check preprocessing pipeline
3. Verify model input requirements
4. Test with known good image

## Development

### Adding Custom Models
1. Convert model to ONNX format
2. Compile for Hailo: `hailo compiler model.onnx --hef model.hef`
3. Update pipeline configuration
4. Test inference performance

### Custom Post-processing
```python
def custom_postprocess(raw_outputs):
    # Parse raw model outputs
    detections = parse_yolo_outputs(raw_outputs)
    
    # Apply custom filtering
    filtered = apply_custom_filters(detections)
    
    return filtered
```

## Integration Points

The tracking module integrates with:
- **Flask Server**: HTTP API endpoints for detection
- **Video Streaming**: Real-time object overlay
- **Robot Control**: Object-based navigation
- **Audio System**: Audio cues for detected objects
- **Light Bar**: Visual feedback for detection events