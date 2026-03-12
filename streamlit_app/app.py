import streamlit as st
import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from model_architecture import (
    DCGenerator,
    DCDiscriminator,
    VanillaGenerator,
    VanillaDiscriminator,
    nz,
)

DEVICE = "cpu"

st.set_page_config(page_title="GAN Portrait Demo", layout="wide")

st.title("GAN Portrait Generator & Detector")
st.write("Compare Vanilla GAN and DCGAN on portrait generation and detection.")

# ====================================
# MODEL SELECTOR
# ====================================

model_choice = st.selectbox("Select Model", ["Vanilla GAN", "DCGAN"])

# ====================================
# LOAD MODELS
# ====================================


@st.cache_resource
def load_models(model_choice):

    if model_choice == "Vanilla GAN":
        generator = VanillaGenerator()
        discriminator = VanillaDiscriminator()

        generator.load_state_dict(
            torch.load("models/vanilla_generator.pth", map_location=DEVICE)
        )

        discriminator.load_state_dict(
            torch.load("models/best_vanilla_discriminator.pth", map_location=DEVICE)
        )

    else:
        generator = DCGenerator()
        discriminator = DCDiscriminator()

        generator.load_state_dict(
            torch.load("models/dcgan_generator.pth", map_location=DEVICE)
        )

        discriminator.load_state_dict(
            torch.load("models/best_dcgan_discriminator.pth", map_location=DEVICE)
        )

    generator.eval()
    discriminator.eval()

    return generator, discriminator


generator, discriminator = load_models(model_choice)

st.divider()

# ====================================
# IMAGE GENERATION
# ====================================

st.header("Generate Fake Portraits")

num_images = st.slider(
    "Number of images to generate", min_value=1, max_value=16, value=4
)

if st.button("Generate Images"):
    with torch.no_grad():
        if model_choice == "Vanilla GAN":
            noise = torch.randn(num_images, nz)

        else:
            noise = torch.randn(num_images, nz, 1, 1)

        fake_images = generator(noise)

        # Convert [-1,1] -> [0,1]
        fake_images = fake_images * 0.5 + 0.5
        fake_images = fake_images.cpu().numpy()

    cols = st.columns(4)

    for i in range(num_images):
        img = np.transpose(fake_images[i], (1, 2, 0))

        cols[i % 4].image(img, caption=f"Generated {i + 1}", use_container_width=True)

st.divider()

# ====================================
# REAL vs FAKE DETECTOR
# ====================================

st.header("Real vs Fake Detection")

uploaded_file = st.file_uploader("Upload portrait image", type=["jpg", "jpeg", "png"])

transform = T.Compose(
    [T.Resize((32, 32)), T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", width=200)

    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = discriminator(img_tensor)

        # DCGAN discriminator outputs logits
        confidence = torch.sigmoid(output).item()

    prediction = "Real" if confidence > 0.5 else "Fake"

    st.subheader(f"Prediction: {prediction}")

    st.progress(confidence)

    st.write(f"Confidence Score: {confidence:.4f}")
