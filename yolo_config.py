from fastapi import UploadFile, HTTPException
import cv2
import numpy as np
from ultralytics import YOLO
import base64
import io
from PIL import ImageFont, ImageDraw, Image
from typing import Dict, List, Any
import logging
import os
import subprocess

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Detection color - using blue for YOLOv11
DETECTION_COLOR = (0, 255, 255)

VIDEO_CONFIGS = {
    'max_duration': 45,  # seconds
    'max_file_size': 50 * 1024 * 1024,  # 50MB
    'output_fps': 40,
    'output_codec': 'H264'
}

class YOLODetector:
    def __init__(self):
        self.model = None
        self.load_model()

    def load_model(self):
        """Load YOLOv11 model only"""
        try:
            self.model = YOLO(r'D:\Project\Traffic_signs\models\best_v11.0.1.2.pt')  
            logger.info("YOLOv11 model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading YOLOv11 model: {e}")
            self.model = None
    
    def detect(self, image: np.ndarray) -> Dict[str, Any]:
        """Detect traffic signs using YOLOv11 model"""
        if self.model is None:
            return {
                "success": False,
                "message": "YOLOv11 model chưa được load",
                "detections": [],
                "annotated_image": image
            }
        
        try:
            results = self.model(image)[0]
            detections = []

            # Handle grayscale images
            if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
                annotated_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            else:
                annotated_image = image.copy()

            # Process detections
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                confidence = float(box.conf[0].cpu().numpy()) 
                class_id = int(box.cls[0].cpu().numpy())

                if confidence > 0.3:
                    class_name = self.model.names.get(class_id, f"Unknown_{class_id}")

                    detections.append({
                        'class': str(class_id),
                        'confidence': confidence,
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'name': class_name
                    })

                    # Draw bounding box
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), DETECTION_COLOR, thickness=2)

                    # Draw label
                    label = f"{class_name} {confidence:.2f}"
                    annotated_image = draw_label_with_vietnamese(
                        annotated_image, label, x1, y1 - 20, color=DETECTION_COLOR
                    )
                    
            return {
                'success': True,
                'message': "Detection completed with YOLOv11",
                'detections': detections,
                'annotated_image': annotated_image
            }
        except Exception as e:
            logger.error(f"Error in YOLOv11 detection: {e}")
            return {
                'success': False,
                "message": f"Error detecting with YOLOv11: {str(e)}",
                'detections': [],
                'annotated_image': image
            }


def image_to_base64(image: np.ndarray) -> str:
    """Convert image to base64 string"""
    try:
        _, buffer = cv2.imencode('.jpg', image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{image_base64}"
    except Exception as e:
        logger.error(f"Error converting image to base64: {e}")
        return ""


def validate_image(file: UploadFile) -> np.ndarray:
    """Validate and convert uploaded image file to numpy array"""
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        contents = file.file.read()
        image = Image.open(io.BytesIO(contents))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        return opencv_image
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")


def validate_video(file: UploadFile, temp_dir: str) -> str:
    """Validate uploaded video file and save to temporary directory"""
    if not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    if file.size > VIDEO_CONFIGS['max_file_size']:
        max_size_mb = VIDEO_CONFIGS['max_file_size'] // 1024 // 1024
        raise HTTPException(status_code=400, detail=f"Video file too large. Maximum size is {max_size_mb}MB")
    
    try:
        video_path = os.path.join(temp_dir, f"input_{file.filename}")
        with open(video_path, 'wb') as buffer:
            contents = file.file.read()
            buffer.write(contents)

        # Check video duration
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps if fps > 0 else 0
        cap.release() 

        if duration > VIDEO_CONFIGS['max_duration']:
            raise HTTPException(
                status_code=400,
                detail=f"Video too long. Maximum duration is {VIDEO_CONFIGS['max_duration']} seconds"
            )
        return video_path
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid video file: {str(e)}")


def process_video(video_path: str, detector: YOLODetector, temp_dir: str) -> Dict[str, Any]:
    """Process video file and detect traffic signs using YOLOv11"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {
                'success': False,
                'message': 'Could not open video file',
                'detections': [],
                'output_path': None
            }
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Set up video writer
        output_path = os.path.join(temp_dir, "output_yolov11.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        unique_detections = {}
        frame_count = 0

        logger.info(f"Processing video with YOLOv11: {total_frames} frames")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            frame_result = detector.detect(frame)

            if frame_result['success']:
                for detection in frame_result['detections']:
                    class_id = detection['class']
                    if class_id not in unique_detections:
                        unique_detections[class_id] = detection
                annotated_frame = frame_result['annotated_image']
            else:
                annotated_frame = frame
            
            out.write(annotated_frame)

            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                logger.info(f"Processing YOLOv11: {progress:.1f}% complete")
        
        cap.release()
        out.release() 

        final_output_path = os.path.join(temp_dir, "output_h264_yolov11.mp4")
        try:
            subprocess.run([
                'ffmpeg', '-i', output_path, '-c:v', 'libx264', '-preset', 'fast',
                '-c:a', 'aac', final_output_path
            ], check=True)
            os.remove(output_path)
            output_path = final_output_path
        except Exception as e:
            logger.warning(f"FFmpeg conversion failed, using original output: {e}")

        final_detections = list(unique_detections.values())
        logger.info(f"Video processing completed: {len(final_detections)} unique detections")

        return {
            'success': True,
            'message': 'Video processing completed with YOLOv11',
            'detections': final_detections,
            'output_path': output_path
        }
        
    except Exception as e:
        logger.error(f"Error processing video with YOLOv11: {e}")
        return {
            'success': False,
            'message': f'Error processing video with YOLOv11: {str(e)}',
            'detections': [],
            'output_path': None
        }


def draw_label_with_vietnamese(image: np.ndarray, text: str, x: int, y: int, color):
    """Draw Vietnamese text label on image"""
    try:
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image_pil)
        
        try:
            font = ImageFont.truetype(r"c:\WINDOWS\Fonts\ARILN.TTF", 24)
        except:
            font = ImageFont.load_default()
            
        color_rgb = (color[2], color[1], color[0])
        draw.text((x, y), text=text, font=font, fill=color_rgb)
    
        return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    
    except Exception as e:
        logger.warning(f"Error drawing Vietnamese text, using OpenCV fallback: {e}")
        cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 3)
        return image
