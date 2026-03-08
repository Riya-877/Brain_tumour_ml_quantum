import tensorflow as tf
import numpy as np
import joblib
from sklearn.metrics import classification_report, accuracy_score

IMG_SIZE = 224
BATCH_SIZE = 16

# Load test dataset
test_ds = tf.keras.utils.image_dataset_from_directory(
    "data/test",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Store true labels
y_true = np.concatenate([y for x, y in test_ds], axis=0)

# ---------------------------
# 1️⃣ Evaluate CNN model
# ---------------------------

cnn_model = tf.keras.models.load_model("models/cnn_model.keras")

cnn_pred = cnn_model.predict(test_ds)
cnn_pred = np.argmax(cnn_pred, axis=1)

print("\nCNN MODEL RESULTS")
print("Accuracy:", accuracy_score(y_true, cnn_pred))
print(classification_report(y_true, cnn_pred))


# ---------------------------
# 2️⃣ Evaluate ResNet model
# ---------------------------

resnet_model = tf.keras.models.load_model("models/resnet_model.keras")

resnet_pred = resnet_model.predict(test_ds)
resnet_pred = np.argmax(resnet_pred, axis=1)

print("\nRESNET MODEL RESULTS")
print("Accuracy:", accuracy_score(y_true, resnet_pred))
print(classification_report(y_true, resnet_pred))


# ---------------------------
# 3️⃣ Feature Extraction
# ---------------------------

base_model = tf.keras.applications.ResNet50(
    include_top=False,
    input_shape=(224,224,3),
    weights="imagenet"
)

feature_extractor = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D()
])

X_test = []

for images, labels in test_ds:
    features = feature_extractor(images)
    X_test.append(features.numpy())

X_test = np.concatenate(X_test)

# ---------------------------
# 4️⃣ Evaluate SVM
# ---------------------------

svm = joblib.load("models/svm_model.pkl")

svm_pred = svm.predict(X_test)

print("\nSVM MODEL RESULTS")
print("Accuracy:", accuracy_score(y_true, svm_pred))
print(classification_report(y_true, svm_pred))


# ---------------------------
# 5️⃣ Evaluate Random Forest
# ---------------------------

rf = joblib.load("models/random_forest.pkl")

rf_pred = rf.predict(X_test)

print("\nRANDOM FOREST RESULTS")
print("Accuracy:", accuracy_score(y_true, rf_pred))
print(classification_report(y_true, rf_pred))