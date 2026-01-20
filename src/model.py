from typing import List

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (Input, BatchNormalization, GlobalAveragePooling2D,
                                     Dense, Multiply, Conv2D, Flatten, Dropout)
from tensorflow.keras.applications import VGG16, ResNet50, DenseNet121
import keras


@keras.saving.register_keras_serializable(package="DR_Detection")
class ModelFusionLayer(tf.keras.layers.Layer):
    """Custom layer that learns a unique weight for each feature channel across the 3 models.

    Expects a list of 3 tensors with identical spatial shape (batch, h, w, channels).
    Internally stacks them into shape (batch, h, w, models, channels) and applies
    learnable per-(model,channel) weights followed by a sum across models.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        # input_shape: list of three shapes
        if not isinstance(input_shape, list) or len(input_shape) != 3:
            raise ValueError("ModelFusionLayer requires a list of 3 tensors as input")
        _, h, w, c = input_shape[0]
        self.models = len(input_shape)
        self.channels = c
        # weight per-model-per-channel
        self.w = self.add_weight(name="fusion_weights",
                                 shape=(self.models, self.channels),
                                 initializer=tf.keras.initializers.Ones(),
                                 trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        # stack inputs: shape -> (batch, h, w, models, channels)
        x = tf.stack(inputs, axis=-2)
        # expand weights to (1,1,1,models,channels)
        w = tf.reshape(self.w, (1, 1, 1, self.models, self.channels))
        weighted = x * w
        fused = tf.reduce_sum(weighted, axis=-2)  # sum over models -> (batch,h,w,channels)
        return fused


def attention_block(x):
    """Simple channel attention block as described in the spec.

    BN -> GAP -> FC -> FC -> Multiply
    """
    from tensorflow.keras.layers import Reshape
    channels = x.shape[-1]
    bn = BatchNormalization()(x)
    gap = GlobalAveragePooling2D()(bn)
    # bottleneck
    hidden = max(16, channels // 16)
    fc1 = Dense(hidden, activation="relu")(gap)
    fc2 = Dense(channels, activation="sigmoid")(fc1)
    scale = Reshape((1, 1, channels))(fc2)
    out = Multiply()([x, scale])
    return out


def build_fusion_model(input_shape=(224, 224, 3), num_classes=5):
    inp = Input(shape=input_shape)

    # Base models (include_top=False). Instantiate with unique names to avoid
    # duplicate layer name collisions when composing multiple pretrained nets.
    vgg_base = VGG16(weights="imagenet", include_top=False, name="vgg_base")
    res_base = ResNet50(weights="imagenet", include_top=False, name="resnet_base")
    dnet_base = DenseNet121(weights="imagenet", include_top=False, name="densenet_base")

    # Freeze base weights
    for m in (vgg_base, res_base, dnet_base):
        m.trainable = False

    # Apply each base model to the shared input tensor
    f1 = vgg_base(inp)
    f2 = res_base(inp)
    f3 = dnet_base(inp)

    # Attention blocks
    a1 = attention_block(f1)
    a2 = attention_block(f2)
    a3 = attention_block(f3)

    # Project each to 512 filters with 1x1 conv
    p1 = Conv2D(512, kernel_size=1, activation="relu", padding="same")(a1)
    p2 = Conv2D(512, kernel_size=1, activation="relu", padding="same")(a2)
    p3 = Conv2D(512, kernel_size=1, activation="relu", padding="same")(a3)

    # Ensure the spatial dims are identical; resize if necessary to the smallest
    # Compute spatial dims dynamically in a lambda-like manner by applying a Conv2D if needed.
    # For simplicity, we will up/down-sample p2 and p3 to p1's spatial dims using tf.image.resize
    _tf = tf

    def _resize_to(reference, var):
        ref_shape = _tf.shape(reference)[1:3]
        var_resized = _tf.image.resize(var, size=ref_shape, method="bilinear")
        return var_resized

    p2r = _tf.keras.layers.Lambda(lambda args: _resize_to(args[0], args[1]))([p1, p2])
    p3r = _tf.keras.layers.Lambda(lambda args: _resize_to(args[0], args[1]))([p1, p3])

    # Fusion: custom layer expects list of three tensors
    fused = ModelFusionLayer()([p1, p2r, p3r])

    flat = Flatten()(fused)
    dense = Dense(256, activation="relu")(flat)
    drop = Dropout(0.5)(dense)
    out = Dense(num_classes, activation="softmax")(drop)

    model = Model(inputs=inp, outputs=out, name="fusion_dr_model")
    return model


if __name__ == "__main__":
    m = build_fusion_model()
    m.summary()
