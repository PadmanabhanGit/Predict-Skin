import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Skin Cancer Prediction - InceptionV3", layout="wide")
st.title("🧠 Skin Cancer Prediction with InceptionV3")
st.markdown("Upload a image to test for prediction and Grad-CAM.")

class_labels = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

@st.cache_resource
def load_model_once():
    return load_model("inceptionv3_skin_cancer.h5")

model = load_model_once()

uploaded_img = st.file_uploader("Upload a skin image", type=["jpg", "jpeg", "png"])

if uploaded_img is not None:
    img_path = "uploaded_image.jpg"
    with open(img_path, "wb") as f:
        f.write(uploaded_img.read())

    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.0

    pred = model.predict(x)
    pred_class = np.argmax(pred)
    predicted_label = class_labels[pred_class]

    st.success(f"🎯 Predicted Class: **{predicted_label}**")

    st.subheader("Prediction Probabilities")
    for i, prob in enumerate(pred[0]):
        st.write(f"- {class_labels[i]}: {prob:.4f}")

    # Grad-CAM
    def get_last_conv_layer(m):
        for layer in reversed(m.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                return layer.name
        return None

    layer_name = get_last_conv_layer(model)
    grad_model = Model([model.inputs], [model.get_layer(layer_name).output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(x)
        loss = predictions[:, pred_class]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0].numpy()
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]

    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    original = cv2.imread(img_path)
    original = cv2.resize(original, (224, 224))
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

    st.subheader("Grad-CAM Visualization")
    col1, col2 = st.columns(2)

    with col1:
        st.image(original, caption="Original Image", channels="BGR", use_column_width=True)
    with col2:
        st.image(overlay, caption="Model Focus (Grad-CAM)", channels="BGR", use_column_width=True)
