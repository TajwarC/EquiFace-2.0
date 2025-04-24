import os
import cv2
import numpy as np
import urllib.request
from ultralytics import YOLO
import tensorflow.lite as tflite

# Constants
YOLO_MODEL_URL = "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov11n-face.pt"
YOLO_MODEL_DIR = os.path.expanduser("~/.cache/equiface")
DEFAULT_YOLO_MODEL_PATH = os.path.join(YOLO_MODEL_DIR, "yolov11n-face.pt")

def _download_yolo_model():
    os.makedirs(YOLO_MODEL_DIR, exist_ok=True)
    if not os.path.exists(DEFAULT_YOLO_MODEL_PATH):
        print(f"Downloading YOLO model to {DEFAULT_YOLO_MODEL_PATH}...")
        urllib.request.urlretrieve(YOLO_MODEL_URL, DEFAULT_YOLO_MODEL_PATH)
        print("Download complete.")

# Load YOLO model once and reuse
_download_yolo_model()
_yolo_model = YOLO(DEFAULT_YOLO_MODEL_PATH)

def preprocess_image(image_path, image_size):
    """
    Loads an image, checks if it contains a face using YOLOv11, and returns a preprocessed image.

    Args:
        image_path (str): Path to the image file.
        image_size (tuple): Tuple of (width, height) for resizing.

    Returns:
        np.ndarray or None: Preprocessed image or None if no face is detected.
    """
    results = _yolo_model(image_path)
    padding = image_size/10
    for result in results:
        boxes = result.boxes
        clss = result.boxes.cls 
    for cls in clss:
        if _yolo_model.names[int(cls)] == 'face':
            image = cv2.imread(image_path)
            boxes = results[0].boxes.xyxy.tolist()
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box
            crop_image = image[int(y1-padding):int(y2+padding), int(x1-padding):int(x2+padding)]
            crop_image = cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB)
            crop_image = cv2.resize(crop_image, (image_size, image_size))
            crop_image = np.expand_dims(crop_image, axis=0).astype(np.float32) / 255.0
            return crop_image
        if _yolo_model.names[int(cls)] != 'face':
            return None 

def get_embedding(interpreter, image):
    """
    Runs inference on a single image using a TFLite interpreter and returns the embedding vector.

    Args:
        interpreter: Loaded TFLite Interpreter.
        image (np.ndarray): Preprocessed input image.

    Returns:
        np.ndarray: Embedding vector.
    """
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index']).flatten()
