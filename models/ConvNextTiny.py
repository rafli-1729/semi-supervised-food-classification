import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.applications import ConvNeXtTiny
from tensorflow.keras.applications.convnext import preprocess_input

pretrained_convnext = ConvNeXtTiny(
    include_top=False,
    weights='imagenet',
    pooling=None,
    input_shape=(*IMG_SIZE, 3)
)
pretrained_convnext.trainable = False

# Input
inputs = keras.Input(shape=(*IMG_SIZE, 3))
x = layers.RandomFlip('horizontal')(inputs)
x = layers.RandomRotation(0.15)(x)
x = layers.RandomZoom(height_factor=(0.0, 0.3),
                      width_factor=(0.0, 0.3))(x)
x = layers.RandomContrast(0.3)(x)
x = layers.RandomBrightness(factor=0.1,
                            value_range=(0, 255))(x)
x = layers.RandomTranslation(0.1, 0.1)(x)
x = layers.RandomColorJitter(0.2)(x)
x = keras_cv.layers.RandomCutout(height_factor=0.4,
                                 width_factor=0.4)(x)

# Preprocessing
x = preprocess_input(x)

# Feature extractor
x = pretrained_convnext(x, training=False)
x = layers.GlobalAveragePooling2D()(x) 

# Head
x = layers.Dense(
    256, activation='swish',
    kernel_regularizer=regularizers.l2(1e-4)
)(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(
    128, activation='gelu',
    kernel_regularizer=regularizers.l2(1e-4)
)(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(15, activation='softmax', dtype='float32')(x)

model_convnext = keras.Model(inputs, outputs)
