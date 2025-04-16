import cv2
import numpy as np
from ultralytics import YOLO
import tensorflow.lite as tflite

# Load YOLO model once and reuse it
_yolo_model = YOLO("yolo11n.pt")

def preprocess_image(image_path):
    """
    Loads an image, checks if it contains a person using YOLO11
    and returns a preprocessed image ready for model input.

    Args:
        image_path (str): Path to the image file.

    Returns:
        np.ndarray or None: Preprocessed image or None if no person detected.
    """
    model = YOLO("yolov11n-face.pt")
    results = model(image_path)
    for result in results:
        boxes = result.boxes
        clss = result.boxes.cls 
    for cls in clss:
        if model.names[int(cls)] == 'face':
            image = cv2.imread(image_path)
            boxes = results[0].boxes.xyxy.tolist()
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box
            # Crop the object using the bounding box coordinates
            crop_image = image[int(y1):int(y2), int(x1):int(x2)]
            #if image is None:
                #return None
            crop_image = cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB)
            crop_image = cv2.resize(crop_image, (224, 224))
            crop_image = np.expand_dims(crop_image, axis=0).astype(np.float32) / 255.0
            return crop_image
        if model.names[int(cls)] != 'face':
            return None 

def get_embedding(interpreter, image):
    """
    Runs inference on a single image using a TFLite interpreter
    and returns the embedding vector.

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
