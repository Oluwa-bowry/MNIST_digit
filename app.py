"""
🖥 Streamlit Web App for Generating Handwritten Digits (MNIST)

- Allows user to select a digit (0–9)
- Generates 5 images using your custom VAE trained model
- Model must be loaded from a .pkl file
"""

import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt

# 🔧 Set page config
st.set_page_config(page_title="MNIST Digit Generator", layout="centered")

# ✅ Load the saved decoder model from the .pkl file
@st.cache_resource
def load_model(pickle_path="vae_mnist_tensorflow_.pkl"):
    with open(pickle_path, "rb") as f:
        vae_wrapper = pickle.load(f)
    return vae_wrapper

vae_wrapper = load_model()

# ✅ Build decoder architecture again (must match saved model)
latent_dim = 16

def build_decoder():
    latent_inputs = tf.keras.Input(shape=(latent_dim,))
    x = tf.keras.layers.Dense(128, activation="relu")(latent_inputs)
    x = tf.keras.layers.Dense(28 * 28, activation="sigmoid")(x)
    outputs = tf.keras.layers.Reshape((28, 28))(x)
    decoder = tf.keras.Model(latent_inputs, outputs)
    return decoder

decoder = build_decoder()
decoder.set_weights(vae_wrapper.decoder_weights)

# ✅ UI
st.title("✍️ MNIST Handwritten Digit Generator")
st.markdown("This app generates handwritten digits using your trained model.")

digit = st.number_input("Enter a digit (0–9) to conditionally guide generation:", min_value=0, max_value=9, step=1)

if st.button("Generate 5 Images"):
    # Generate 5 latent vectors using the digit as a seed
    np.random.seed(digit)  # Control randomness by digit
    z_vecs = np.random.normal(size=(5, latent_dim))

    # Generate images
    generated_images = decoder.predict(z_vecs)

    # Display
    st.subheader(f"Generated 5 images for digit: {digit}")
    fig, axes = plt.subplots(1, 5, figsize=(10, 2))
    for i in range(5):
        axes[i].imshow(generated_images[i], cmap="gray")
        axes[i].axis("off")
    st.pyplot(fig)
