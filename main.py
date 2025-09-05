import logging
import os
import time
import cv2
import subprocess
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, File, UploadFile, HTTPException
from yolo_config import YOLODetector, image_to_base64, validate_image, validate_video, process_video
from pydantic import BaseModel
from typing import Dict, List, Any, AsyncGenerator
import tempfile
import shutil
from io import BytesIO
import requests
from requests.exceptions import RequestException, HTTPError, JSONDecodeError
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Traffic Sign Detection API - YOLOv11",
    description="API for Vietnamese traffic sign detection using YOLOv11 with video support",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

detector = YOLODetector()

class AudioRequest(BaseModel):
    detections: List[Dict[str, Any]]
    file_type: str

async def stream_video_frames(video_path: str, detector: YOLODetector) -> AsyncGenerator[bytes, None]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Could not open video file for streaming")
        return
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_result = detector.detect(frame)
            annotated_frame = frame_result['annotated_image'] if frame_result['success'] else frame
            
            # Encode frame to JPEG for streaming
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n'
            
            await asyncio.sleep(0.033)  # Approximate 30 fps
            
    except Exception as e:
        logger.error(f"Error streaming video frames: {e}")
    finally:
        cap.release()

@app.get("/")
async def serve_frontend():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    else:
        return {
            "message": "Traffic Sign Detection API - YOLOv11", 
            "version": "2.0.0", 
            "frontend": "Place index.html file in the same directory"
        }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "model_loaded": detector.model is not None,
        "model": "YOLOv11"
    }

@app.post("/api/detect/image")
async def detect_image(file: UploadFile = File(...)):
    try:
        image = validate_image(file)
        result = detector.detect(image)

        if result['success']:
            annotated_image_base64 = image_to_base64(result['annotated_image'])
        else:
            annotated_image_base64 = image_to_base64(image)

        return JSONResponse(content={
            "success": result['success'],
            "message": result['message'],
            "file_type": "image",
            "model": "YOLOv11",
            "detections": result['detections'],
            "annotated_media": annotated_image_base64,
            "total_detections": len(result['detections'])
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing image: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/api/detect/video")
async def detect_video(file: UploadFile = File(...)):
    temp_dir = None
    try:
        temp_dir = tempfile.mkdtemp()
        video_path = validate_video(file, temp_dir=temp_dir)

        # Process video to save complete output
        result = process_video(video_path=video_path, detector=detector, temp_dir=temp_dir)

        video_url = None
        if result['success'] and result['output_path']:
            static_dir = "static/videos"
            os.makedirs(static_dir, exist_ok=True)
            output_filename = f"output_{int(time.time())}_yolov11.mp4"
            output_path = os.path.join(static_dir, output_filename)
            shutil.move(result['output_path'], output_path)
            video_url = f"/static/videos/{output_filename}"

        return JSONResponse(content={
            "success": result['success'],
            "message": result['message'],
            "file_type": "video",
            "model": "YOLOv11",
            "detections": result['detections'],
            "annotated_media": video_url,
            "total_detections": len(result['detections'])
        })
    
    except HTTPException:
        raise
    except Exception as e: 
        logger.error(f"Error processing video: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

@app.get("/api/stream/video/{filename}")
async def stream_video(filename: str):
    video_path = os.path.join("static/videos", filename)
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video not found")
    
    return StreamingResponse(
        stream_video_frames(video_path, detector),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.post("/api/cleanup/videos")
async def cleanup_videos():
    try:
        static_dir = "static/videos"
        if not os.path.exists(static_dir):
            return JSONResponse(content={"success": True, "message": "No videos to clean"})
        
        current_time = time.time()
        max_age = 3600  # 1 hour
        deleted_files = []
        
        for filename in os.listdir(static_dir):
            file_path = os.path.join(static_dir, filename)
            if os.path.isfile(file_path):
                file_age = current_time - os.path.getmtime(file_path)
                if file_age > max_age:
                    os.remove(file_path)
                    deleted_files.append(filename)
        
        return JSONResponse(content={
            "success": True,
            "message": f"Cleaned {len(deleted_files)} old video(s)",
            "deleted_files": deleted_files
        })
    
    except Exception as e:
        logger.error(f"Error cleaning up videos: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error cleaning videos: {str(e)}")

@app.post("/api/audio/generate")
async def generate_audio(request: AudioRequest):
    try:
        detections = request.detections
        file_type = request.file_type
        unique_detections = {}
        for detection in detections:
            class_id = detection['class']
            if class_id not in unique_detections:
                unique_detections[class_id] = detection

        all_detections = list(unique_detections.values())

        if not all_detections:
            if file_type == 'image':
                text_content = 'Không phát hiện được biển báo giao thông nào trong ảnh này'
            else:
                text_content = 'Không phát hiện được biển báo giao thông nào trong video này'
        else:
            detection_names = [detection['name'] for detection in all_detections]
            if file_type == 'image':
                if len(detection_names) == 1:
                    text_content = f"Phát hiện được biển báo: {detection_names[0]}."
                else:
                    text_content = f"Phát hiện được {len(detection_names)} biển báo gồm: {', '.join(detection_names[:-1])} và {detection_names[-1]}."
            else:
                if len(detection_names) == 1:
                    text_content = f"Trong video phát hiện được biển báo: {detection_names[0]}."
                else:
                    text_content = f"Trong video phát hiện được {len(detection_names)} biển báo gồm: {', '.join(detection_names[:-1])} và {detection_names[-1]}."
        
        api_key = os.getenv("FPT_API_KEYS")
        if not api_key:
            raise ValueError("Missing FPT API key. Please configure the FPT_API_KEYS environment variable.")

        audio_buffer = generate_audio_with_fpt(text_content, api_key, voice="banmai")

        return StreamingResponse(
            BytesIO(audio_buffer),
            media_type="audio/mpeg",
            headers={"Content-Disposition": "attachment; filename=detection_audio.mp3"}
        )
        
    except ValueError as ve:
        logger.error(f"Validation error in audio generation: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re:
        logger.error(f"Runtime error in audio generation: {re}")
        raise HTTPException(status_code=500, detail=str(re))
    except Exception as e:
        logger.error(f"Unexpected error in audio generation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

def generate_audio_with_fpt(text: str, api_key: str, voice: str = "banmai") -> bytes:
    if not api_key:
        raise ValueError("Missing FPT API key")

    url = "https://api.fpt.ai/hmi/tts/v5"
    headers = {
        "api-key": api_key,
        "voice": voice,
    }

    try:
        logger.info("Sending POST request to FPT.AI TTS API")
        response = requests.post(url, data=text.encode("utf-8"), headers=headers, timeout=10)
        response.raise_for_status()

        try:
            result = response.json()
        except JSONDecodeError:
            raise ValueError("Invalid JSON response from FPT.AI")

        if "async" not in result:
            raise ValueError(f"Invalid response format from FPT.AI: {result}")

        audio_url = result["async"]
        logger.info(f"Received async audio URL: {audio_url}")

        max_retries = 5
        retry_delay = 2
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempt {attempt + 1}: Fetching audio from {audio_url}")
                audio_response = requests.get(audio_url, timeout=10)
                audio_response.raise_for_status()
                logger.info("Audio fetched successfully")
                return audio_response.content
            except HTTPError as he:
                status = audio_response.status_code if 'audio_response' in locals() else None
                if status in [404, 503]:
                    if attempt < max_retries - 1:
                        logger.warning(f"Audio not ready (status {status}). Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                raise he
            except RequestException as re:
                logger.error(f"Network error during audio fetch: {re}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                raise re

        raise RuntimeError("Max retries exceeded while fetching audio from FPT.AI")

    except HTTPError as he:
        status = response.status_code if 'response' in locals() else None
        if status == 401:
            raise ValueError("Invalid FPT.AI API key")
        elif status == 429:
            raise RuntimeError("FPT.AI rate limit exceeded")
        elif status == 400:
            raise ValueError("Bad request to FPT.AI (check text input)")
        else:
            raise RuntimeError(f"FPT.AI HTTP error: {he}")
    except RequestException as re:
        raise RuntimeError(f"FPT.AI network error: {re}")
    except ValueError as ve:
        raise ve
    except Exception as e:
        logger.error(f"Unexpected error in FPT.AI integration: {e}", exc_info=True)
        raise RuntimeError(f"Unexpected error: {str(e)}")

@app.get("/api/classes")
async def get_classes():
    return JSONResponse(content={
        "classes": CLASS_MAPPING,
        "total_classes": len(CLASS_MAPPING)
    })

@app.post("/api/detect")
async def detect(file: UploadFile = File(...)):
    if file.content_type.startswith("image/"):
        return await detect_image(file)
    elif file.content_type.startswith("video/"):
        return await detect_video(file)
    else:
        raise HTTPException(status_code=400, detail="Invalid file type. Must be image or video.")


def print_startup_info():
    print("\n" + "="*70)
    print("🚦 TRAFFIC SIGN DETECTION SERVER - YOLOv11 Version")
    print("="*70)
    print("📡 API Server: http://localhost:8000")
    print("🌐 Frontend: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    print("🔍 Health Check: http://localhost:8000/health")
    print("="*70)
    print("🎯 Features:")
    print("   • YOLOv11 Traffic Sign Detection")
    print("   • Image & Video Processing")
    print("   • Real-time Video Streaming")
    print("   • Audio Narration (Vietnamese)")
    print("   • Enhanced UI Design")
    print("="*70)
    print("📋 Instructions:")
    print("   1. Open http://localhost:8000 in your browser")
    print("   2. Upload an image or video (max 45s) to detect traffic signs")
    print("   3. View YOLOv11 detection results in real-time")
    print("   4. Listen to Vietnamese audio narration of detected signs")
    print("="*70)
    print("Press CTRL+C to stop the server\n")

if __name__ == "__main__":
    import uvicorn
    import asyncio
    print_startup_info()
    uvicorn.run(app, host="0.0.0.0", port=8000)