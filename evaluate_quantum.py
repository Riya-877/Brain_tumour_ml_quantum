import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, accuracy_score

IMG_SIZE = 224
BATCH_SIZE = 16

test_ds = tf.keras.utils.image_dataset_from_directory(
    "data/test",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False
)

y_true = np.concatenate([y for x, y in test_ds], axis=0)

model = tf.keras.models.load_model("models/hybrid_model.keras")

pred = model.predict(test_ds)
pred = np.argmax(pred, axis=1)

print("HYBRID QUANTUM MODEL RESULTS")
print("Accuracy:", accuracy_score(y_true, pred))
print(classification_report(y_true, pred))