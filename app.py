import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# ----------------------------
# Config
# ----------------------------
st.set_page_config(
    page_title="Brain Tumor Detection",
    layout="centered"
)

CLASS_NAMES = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]
CONF_THRESHOLD = 0.6

# ----------------------------
# Load CNN model
# ----------------------------
model = tf.keras.models.load_model(
    "models/brain_tumor_model.h5",
    compile=False,
    safe_mode=False
)

# ----------------------------
# Image preprocessing
# ----------------------------
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    img = np.array(image) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ----------------------------
# Grad-CAM
# ----------------------------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_gradcam(image, heatmap, alpha=0.4):
    image_np = np.array(image)

    if image_np.ndim == 3 and image_np.shape[2] == 4:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)

    heatmap = cv2.resize(
        heatmap,
        (image_np.shape[1], image_np.shape[0])
    )

    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    if image_np.dtype != np.uint8:
        image_np = np.uint8(255 * image_np)

    overlay = cv2.addWeighted(
        image_np, 1 - alpha,
        heatmap, alpha,
        0
    )

    return overlay

# ----------------------------
# Quantum validation
# ----------------------------
def quantum_validate(prob):
    qc = QuantumCircuit(1, 1)
    theta = 2 * np.arcsin(np.sqrt(prob))
    qc.ry(theta, 0)
    qc.measure(0, 0)

    sim = AerSimulator()
    compiled = transpile(qc, sim)
    result = sim.run(compiled, shots=1024).result()
    return result.get_counts()

# ----------------------------
# UI
# ----------------------------
st.title("🧠 Brain Tumor Detection")
st.write("Upload a brain MRI image for classification with explainability.")

uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI", use_container_width=True)

    # ----------------------------
    # Prediction
    # ----------------------------
    input_img = preprocess_image(image)
    preds = model.predict(input_img)

    class_index = np.argmax(preds)
    confidence = float(preds[0][class_index])

    if confidence < CONF_THRESHOLD:
        predicted_class = "Uncertain / Likely No Tumor"
        tumor_flag = False
    else:
        predicted_class = CLASS_NAMES[class_index]
        tumor_flag = predicted_class != "No Tumor"

    # ----------------------------
    # Debug (keep for now)
    # ----------------------------
    st.write("🔍 Raw prediction vector:", preds)

    # ----------------------------
    # Output UI
    # ----------------------------
    st.subheader("🧪 CNN Prediction")
    st.success(f"Prediction: {predicted_class}")
    st.info(f"Confidence: {confidence * 100:.2f}%")
    st.progress(int(confidence * 100))

    if tumor_flag:
        st.error("⚠️ Tumor Detected")
    else:
        st.success("✅ No Tumor Detected")

    # ----------------------------
    # Grad-CAM
    # ----------------------------
    heatmap = make_gradcam_heatmap(
        input_img,
        model,
        last_conv_layer_name="conv2d_2",
        pred_index=class_index
    )

    gradcam_img = overlay_gradcam(image, heatmap)

    st.subheader("🔥 Explainable AI (Grad-CAM)")

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            image,
            caption="Original MRI",
            use_container_width=True
        )

    with col2:
        st.image(
            gradcam_img,
            caption="Grad-CAM Heatmap",
            use_container_width=True
        )

    # ----------------------------
    # Probability Bar Chart
    # ----------------------------
    st.subheader("📊 Class-wise Probability Distribution")
    fig, ax = plt.subplots()
    ax.bar(CLASS_NAMES, preds[0])
    ax.set_ylim([0, 1])
    ax.set_ylabel("Probability")
    plt.xticks(rotation=15)
    st.pyplot(fig)

    # ----------------------------
    # Quantum Validation (SAFE)
    # ----------------------------
    st.subheader("⚛️ Quantum Validation")

    if confidence >= CONF_THRESHOLD:
        counts = quantum_validate(confidence)
        st.json(counts)
    else:
        st.info("Quantum validation skipped due to low model confidence.")