import tensorflow as tf
from tensorflow.keras import layers

IMG_SIZE = 224
BATCH_SIZE = 16

train_ds = tf.keras.utils.image_dataset_from_directory(
    "data/train",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "data/test",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

normalization_layer = layers.Rescaling(1./255)

base_model = tf.keras.applications.ResNet50(
    include_top=False,
    input_shape=(224,224,3),
    weights="imagenet"
)

base_model.trainable = False

inputs = tf.keras.Input(shape=(224,224,3))

x = normalization_layer(inputs)
x = base_model(x)

x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation="relu")(x)

outputs = layers.Dense(4, activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

model.save("models/resnet_model.keras")