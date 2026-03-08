import tensorflow as tf
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import joblib

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

base_model = tf.keras.applications.ResNet50(
    include_top=False,
    input_shape=(224,224,3),
    weights="imagenet"
)

feature_extractor = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D()
])

X_train = []
y_train = []

for images, labels in train_ds:
    features = feature_extractor(images)
    X_train.append(features.numpy())
    y_train.append(labels.numpy())

X_train = np.concatenate(X_train)
y_train = np.concatenate(y_train)

# SVM
svm = SVC(kernel='rbf')
svm.fit(X_train, y_train)

joblib.dump(svm, "models/svm_model.pkl")

# Random Forest
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

joblib.dump(rf, "models/random_forest.pkl")

print("Models trained successfully")