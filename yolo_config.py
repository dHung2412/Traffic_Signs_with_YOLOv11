from fastapi import UploadFile, HTTPException
import cv2
import numpy as np
from ultralytics import YOLO
import base64
import io
from PIL import ImageFont, ImageDraw, Image
from typing import Dict, List, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Vietnamese traffic sign class mapping
CLASS_MAPPING = {
    0: "Đường người đi bộ cắt ngang",
    1: "Đường giao nhau (ngã ba bên phải)",
    2: "Cấm đi ngược chiều",
    3: "Phải đi vòng sang bên phải",
    4: "Giao nhau với đường đồng cấp",
    5: "Giao nhau với đường không ưu tiên",
    6: "Chỗ ngoặt nguy hiểm vòng bên trái",
    7: "Cấm rẽ trái",
    8: "Bến xe buýt",
    9: "Nơi giao nhau chạy theo vòng xuyến",
    10: "Cấm dừng và đỗ xe",
    11: "Chỗ quay xe",
    12: "Biển gộp làn đường theo phương tiện",
    13: "Đi chậm",
    14: "Cấm xe tải",
    15: "Đường bị thu hẹp về phía phải",
    16: "Giới hạn chiều cao",
    17: "Cấm quay đầu",
    18: "Cấm ô tô khách và ô tô tải",
    19: "Cấm rẽ phải và quay đầu",
    20: "Cấm ô tô",
    21: "Đường bị thu hẹp về phía trái",
    22: "Gồ giảm tốc phía trước",
    23: "Cấm xe hai và ba bánh",
    24: "Kiểm tra",
    25: "Chỉ dành cho xe máy",
    26: "Chướng ngoại vật phía trước",
    27: "Trẻ em",
    28: "Xe tải và xe công",
    29: "Cấm mô tô và xe máy",
    30: "Chỉ dành cho xe tải",
    31: "Đường có camera giám sát",
    32: "Cấm rẽ phải",
    33: "Nhiều chỗ ngoặt nguy hiểm liên tiếp, chỗ đầu tiên sang phải",
    34: "Cấm xe sơ-mi rơ-moóc",
    35: "Cấm rẽ trái và phải",
    36: "Cấm đi thẳng và rẽ phải",
    37: "Đường giao nhau (ngã ba bên trái)",
    38: "Giới hạn tốc độ (50km/h)",
    39: "Giới hạn tốc độ (60km/h)",
    40: "Giới hạn tốc độ (80km/h)",
    41: "Giới hạn tốc độ (40km/h)",
    42: "Các xe chỉ được rẽ trái",
    43: "Chiều cao tĩnh không thực tế",
    44: "Nguy hiểm khác",
    45: "Đường một chiều",
    46: "Cấm đỗ xe",
    47: "Cấm ô tô quay đầu xe (được rẽ trái)",
    48: "Giao nhau với đường sắt có rào chắn",
    49: "Cấm rẽ trái và quay đầu xe",
    50: "Chỗ ngoặt nguy hiểm vòng bên phải",
    51: "Chú ý chướng ngại vật & vòng tránh sang bên phải"
}

MODEL_COLORS = {
    'yolov8': (0, 255, 0),    # Green
    'yolov11': (255, 0, 0),   # Blue
}

class YOLODetector:
    def __init__(self):
        self.models = {}
        self.load_models()
        self.models_loaded = False

    
    def load_models(self):
        try:
            self.models['yolov8'] = YOLO(r'D:\Project\Traffic_signs\models\v8\best_v8_001.pt')  
            self.models['yolov11'] = YOLO(r'D:\Project\Traffic_signs\models\v11\best_v11_001.pt')  
            logger.info("Models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            pass
    
    def detect(self, image: np.ndarray, model_name: str) -> Dict[str, Any]:
        if self.models is None or model_name not in self.models:
            return {
                "success": False,
                "message": f"Model {model_name} chưa được load",
                "detections": [],
                "annotated_image": image
            }
        
        try:
            model = self.models[model_name]
            results = model(image)[0]
            
            detections = []

            if len(image.shape) == 2 or image.shape[2] == 1: # Ảnh có 1 kênh
                annotated_image = cv2.cvtColor(image, cv2.COLOR_GRAY)

            else:
                annotated_image = image.copy()

            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                confidence = float(box.conf[0].cpu().numpy()) 
                class_id = int(box.cls[0].cpu().numpy())

                if confidence > 0.3:
                    detections.append({
                        'class': str(class_id),
                        'confidence': confidence,
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'name': CLASS_MAPPING.get(class_id, "Unknown")
                    })

                    color = MODEL_COLORS.get(model_name, (0,255,0))
                    cv2.rectangle(annotated_image, pt1=(x1,y1), pt2=(x2,y2), color=color, thickness=2)

                    label = f"{CLASS_MAPPING.get(class_id, class_id)} {confidence:.2f}"
                    annotated_image = draw_label_with_vietnamese(
                        annotated_image, label, x1, y1 -20, color=color
                    )
            return {
                'success': True,
                'message': f"Detection completed with {model_name}",
                'detections': detections,
                'annotated_image': annotated_image
            }
        except Exception as e:
            logger.error(f"Error in detection: {e}")
            return {
                'success': False,
                "message": f"Error detecting with {model_name}: {str(e)}",
                'detection': [],
                'annotated_image': image
            }


def image_to_base64(image: np.ndarray) -> str:

    _, buffer = cv2.imencode('.jpg', image)
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{image_base64}"

def validate_image(file: UploadFile) -> np.ndarray:

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


def draw_label_with_vietnamese(image: np.ndarray, text: str, x: int, y: int, color):
    """
    image: ảnh numpy(BGR)
    text: nội dung tiếng Việt
    (x,y): vị trí về ( góc trái dưới của text)
    """

    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)

    font = ImageFont.truetype(r"c:\WINDOWS\Fonts\ARIALN.TTF", 15)

    color_rgb = (color[2], color[1],color[0])

    draw.text((x,y), text=text, font=font, fill=color_rgb)

    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
