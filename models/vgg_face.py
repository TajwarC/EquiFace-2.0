import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, Sequential
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

# Build the model
functional_model = base_model_functional()
functional_model.load_weights("vgg_face_weights.h5") # Download link "https://github.com/serengil/deepface_models/releases/download/v1.0/vgg_face_weights.h5"

# Extract embeddings from the fc7 layer
embedding_model = Model(inputs=functional_model.input,
                        outputs=functional_model.get_layer("fc7").output)

# Convert the model
converter = tf.lite.TFLiteConverter.from_keras_model(embedding_model)
tflite_model = converter.convert()

# Save the model
with open('vgg_face.tflite', 'wb') as f:
  f.write(tflite_model)