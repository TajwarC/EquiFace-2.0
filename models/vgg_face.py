import os
import argparse
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, ZeroPadding2D, Convolution2D, MaxPooling2D
from tensorflow.keras.layers import Dropout, Flatten, Activation

"""
Creating the VGG Face model from its weights and converting into the TFLite format.
Most of the code is taken from https://github.com/serengil/deepface/blob/master/deepface/models/facial_recognition/VGGFace.py
"""

def base_model_functional():
    input_layer = Input(shape=(224, 224, 3))
    x = ZeroPadding2D((1, 1))(input_layer)
    x = Convolution2D(64, (3, 3), activation="relu")(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Convolution2D(64, (3, 3), activation="relu")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = ZeroPadding2D((1, 1))(x)
    x = Convolution2D(128, (3, 3), activation="relu")(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Convolution2D(128, (3, 3), activation="relu")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = ZeroPadding2D((1, 1))(x)
    x = Convolution2D(256, (3, 3), activation="relu")(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Convolution2D(256, (3, 3), activation="relu")(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Convolution2D(256, (3, 3), activation="relu")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = ZeroPadding2D((1, 1))(x)
    x = Convolution2D(512, (3, 3), activation="relu")(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Convolution2D(512, (3, 3), activation="relu")(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Convolution2D(512, (3, 3), activation="relu")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = ZeroPadding2D((1, 1))(x)
    x = Convolution2D(512, (3, 3), activation="relu")(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Convolution2D(512, (3, 3), activation="relu")(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Convolution2D(512, (3, 3), activation="relu")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Convolution2D(4096, (7, 7), activation="relu", name="fc6")(x)
    x = Dropout(0.5)(x)
    x = Convolution2D(4096, (1, 1), activation="relu", name="fc7")(x)
    x = Dropout(0.5)(x)
    x = Convolution2D(2622, (1, 1), name="predictions")(x)
    x = Flatten()(x)
    output = Activation("softmax")(x)

    model = Model(inputs=input_layer, outputs=output)
    return model

def convert_vgg_face(weights_path, output_path):
    print(f"Loading weights from {weights_path}...")
    functional_model = base_model_functional()
    functional_model.load_weights(weights_path)

    print("Extracting embeddings from the fc7 layer...")
    embedding_model = Model(inputs=functional_model.input,
                            outputs=functional_model.get_layer("fc7").output)

    print("Converting the model to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(embedding_model)
    tflite_model = converter.convert()

    print(f"Saving the model to {output_path}...")
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    print("Conversion complete.")

def main():
    default_weights = Path(__file__).parent / "vgg_face_weights.h5"

    parser = argparse.ArgumentParser(description="Convert VGG Face Keras model to TFLite")
    parser.add_argument("--weights", type=str, default=str(default_weights),
                        help=f"Path to vgg_face_weights.h5 (default: {default_weights})")
    parser.add_argument("--output", type=str, default="vgg_face.tflite",
                        help="Path to save the output TFLite model (default: vgg_face.tflite)")

    args = parser.parse_args()

    if not os.path.exists(args.weights):
        print(f"Error: Weights file not found at {args.weights}")
        print("Download link: https://github.com/serengil/deepface_models/releases/download/v1.0/vgg_face_weights.h5")
        return

    convert_vgg_face(args.weights, args.output)

if __name__ == "__main__":
    main()