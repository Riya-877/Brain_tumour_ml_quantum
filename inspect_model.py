import tensorflow as tf

# Load model
model = tf.keras.models.load_model("models/brain_tumor_model.keras")

# Basic Info
print("\n===== MODEL INPUT SHAPE =====")
print(model.input_shape)

print("\n===== MODEL OUTPUT SHAPE =====")
print(model.output_shape)

print("\n===== MODEL SUMMARY =====")
model.summary()

print("\n===== OUTPUT ACTIVATION =====")
print(model.layers[-1].activation)

print("\n===== NUMBER OF CLASSES =====")
print(model.output_shape[-1])
