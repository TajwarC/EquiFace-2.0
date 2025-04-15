# image_utils.py

import cv2
import numpy as np
from ultralytics import YOLO
import tensorflow.lite as tflite

# Load YOLO model once and reuse it
_yolo_model = YOLO("yolo11n.pt")

def preprocess_image(image_path):
    """
    Loads an image, checks if it contains a person using YOLO,
    and returns a preprocessed image ready for model input.

    Args:
        image_path (str): Path to the image file.

    Returns:
        np.ndarray or None: Preprocessed image or None if no person detected.
    """
    results = _yolo_model(image_path)
    for result in results:
        for cls in result.boxes.cls:
            if _yolo_model.names[int(cls)] == 'person':
                image = cv2.imread(image_path)
                if image is None:
                    return None
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (160, 160))
                image = np.expand_dims(image, axis=0).astype(np.float32) / 255.0
                return image
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
