import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

pretrained_effnet = EfficientNetV2S(
    include_top=False,
    weights='imagenet',
    pooling=None,
    input_shape=(*IMG_SIZE, 3)
)
pretrained_effnet.trainable = False

# Input
inputs = keras.Input(shape=(*IMG_SIZE, 3))

# Augmentasi
x = layers.RandomFlip('horizontal')(inputs)
x = layers.RandomRotation(0.1)(x)
x = layers.RandomZoom(0.1)(x)
x = layers.RandomContrast(0.1)(x)
x = layers.RandomBrightness(factor=0.0001)(x)
x = layers.RandomTranslation(0.1, 0.1)(x)
x = layers.RandomColorJitter(0.1)(x)
x = keras_cv.layers.RandomCutout(height_factor=0.5,
                                 width_factor=0.5)(x)

# Preprocessing
x = preprocess_input(x)

# Feature extractor
x = pretrained_effnet(x, training=False)
x = layers.GlobalAveragePooling2D()(x) 

# Head
x = layers.Dense(
    64, activation='relu',
    kernel_regularizer=regularizers.l2(1e-3)
)(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(15, activation='softmax', dtype='float32')(x)

model_effnet = keras.Model(inputs, outputs)
