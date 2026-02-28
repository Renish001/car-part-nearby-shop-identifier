import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from PIL import Image
import os

# ---------------------------------
# CONFIG
# ---------------------------------
st.set_page_config(page_title="Car Part Identifier", layout="centered")
st.title("🚗 Car Part Identifier System")

DATA_DIR = "dataset"
MODEL_PATH = "car_model.pth"
BATCH_SIZE = 16
EPOCHS = 10
LR = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------
# TRANSFORM (USED FOR BOTH TRAIN & TEST)
# ---------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# ---------------------------------
# TRAIN MODEL IF NOT EXISTS
# ---------------------------------
def train_model():

    dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    classes = dataset.classes

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    st.info("Training model... Please wait")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0

        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        st.write(f"Epoch {epoch+1}/{EPOCHS} - Loss: {running_loss/len(loader):.4f}")

    torch.save({
        "model_state_dict": model.state_dict(),
        "classes": classes
    }, MODEL_PATH)

    st.success("Model trained and saved successfully!")

    return model, classes


# ---------------------------------
# LOAD MODEL
# ---------------------------------
@st.cache_resource
def load_or_train():

    if not os.path.exists(DATA_DIR):
        st.error("Dataset folder not found!")
        st.stop()

    if not os.path.exists(MODEL_PATH):
        return train_model()

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    classes = checkpoint["classes"]

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, classes


model, classes = load_or_train()

# ---------------------------------
# IMAGE UPLOAD
# ---------------------------------
uploaded_file = st.file_uploader("Upload Car Part Image", type=["jpg", "jpeg", "png"])

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_tensor = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_index = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_index].item()

    predicted_class = classes[predicted_index]

    st.success(f"Predicted: {predicted_class}")
    st.info(f"Confidence: {confidence*100:.2f}%")