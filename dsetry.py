import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os

st.set_page_config(page_title="Car Part Identifier", layout="centered")
st.title("🚗 Car Part Identifier System")

# ----------------------------------
# LOAD MODEL
# ----------------------------------
@st.cache_resource
def load_model():

    if not os.path.exists("car_model.pth"):
        st.error("Model file not found. Train model first.")
        st.stop()

    checkpoint = torch.load("car_model.pth", map_location="cpu")

    classes = checkpoint["classes"]

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, classes


model, classes = load_model()

# ----------------------------------
# TRANSFORM (MUST MATCH TRAINING)
# ----------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# ----------------------------------
# IMAGE UPLOAD
# ----------------------------------
uploaded_file = st.file_uploader("Upload Car Part Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_index = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_index].item()

    predicted_class = classes[predicted_index]

    st.success(f"Predicted Car Part: {predicted_class}")
    st.info(f"Confidence Level: {confidence*100:.2f}%")