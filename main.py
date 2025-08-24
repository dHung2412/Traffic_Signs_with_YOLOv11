import logging
import os
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, File, UploadFile, HTTPException
from yolo_config import YOLODetector, CLASS_MAPPING
from yolo_config import image_to_base64, validate_image


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Traffic Sign Detection API",
    description="API for Vietnamese traffic sign detection using YOLOv8 and YOLOv11",
    version="1.0.0"
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

@app.get("/")
async def serve_frontend():
    """Serve the main HTML frontend"""
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    else:
        return {"message": "Traffic Sign Detection API", "version": "1.0.0", "frontend": "Place index.html file in the same directory"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "models_loaded": detector.models is not None}

@app.post("/api/detect/both")
async def detect_both_models(file: UploadFile = File(...)):
    """
    Detect traffic signs using both YOLOv8 and YOLOv11 models
    
    Args:
        file: Image file to process
    
    Returns:
        JSON response with results from both models
    """
    try:
        # Validate and process image
        image = validate_image(file)
        
        # Perform detection with both models
        yolov8_result = detector.detect(image, 'yolov8')
        yolov11_result = detector.detect(image, 'yolov11')
        
        # Convert annotated images to base64
        yolov8_image_base64 = image_to_base64(yolov8_result['annotated_image'])
        yolov11_image_base64 = image_to_base64(yolov11_result['annotated_image'])
        
        return JSONResponse(content={
            "success": True,
            "yolov8": {
                "success": yolov8_result.get('success', True),
                "message": yolov8_result.get('message', ''),
                "detections": yolov8_result['detections'],
                "annotated_image": yolov8_image_base64,
                "total_detections": len(yolov8_result['detections'])
            },
            "yolov11": {
                "success": yolov11_result.get('success', True),
                "message": yolov11_result.get('message', ''),
                "detections": yolov11_result['detections'],
                "annotated_image": yolov11_image_base64,
                "total_detections": len(yolov11_result['detections'])
            }
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/api/classes")
async def get_classes():
    """Get list of traffic sign classes"""
    return JSONResponse(content={
        "classes": CLASS_MAPPING,
        "total_classes": len(CLASS_MAPPING)
    })

def print_startup_info():
    """Print startup information with URLs"""
    print("\n" + "="*60)
    print("🚦 TRAFFIC SIGN DETECTION SERVER")
    print("="*60)
    print("📡 API Server: http://localhost:8000")
    print("🌐 Frontend: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    print("🔍 Health Check: http://localhost:8000/health")
    print("="*60)
    print("📝 Instructions:")
    print("   1. Open http://localhost:8000 in your browser")
    print("   2. Upload an image to detect traffic signs")
    print("   3. View results from both YOLOv8 and YOLOv11")
    print("="*60)
    print("Press CTRL+C to stop the server\n")

if __name__ == "__main__":
    import uvicorn
    print_startup_info()
    uvicorn.run(app, host="0.0.0.0", port=8000)