import tensorflow as tf
from tensorflow.keras import layers, models, applications

IMG_SHAPE = (224, 224, 3)

def build_model():
    # 1. Base model: EfficientNetB0
    base_model = applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=IMG_SHAPE
    )
    base_model.trainable = False   # freeze backbone

    # 2. Add classification head
    inputs = layers.Input(shape=IMG_SHAPE)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(5, activation="softmax")(x)

    model = models.Model(inputs, outputs)

    # 3. Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model, base_model
