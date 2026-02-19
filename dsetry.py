import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import torch.nn.functional as F
import pandas as pd
from PIL import Image
import os
from geopy.distance import geodesic
from geopy.geocoders import Nominatim

# ==========================================
# Base Paths
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PARTS_CSV = os.path.join(BASE_DIR, "car parts.csv")
SHOPS_CSV = os.path.join(BASE_DIR, "coimbatore_spare_parts_shops.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model.pth")

# ==========================================
# Validate Required Files
# ==========================================
for path, name in [
    (PARTS_CSV, "car parts.csv"),
    (SHOPS_CSV, "coimbatore_spare_parts_shops.csv"),
    (MODEL_PATH, "model.pth")
]:
    if not os.path.exists(path):
        st.error(f"{name} not found.")
        st.stop()

# ==========================================
# Load Checkpoint FIRST (to get classes)
# ==========================================
checkpoint = torch.load(MODEL_PATH, map_location="cpu")

if isinstance(checkpoint, dict) and "classes" in checkpoint:
    classes = checkpoint["classes"]
else:
    # Fallback: Load from CSV safely
    df = pd.read_csv(PARTS_CSV)

    if "labels" not in df.columns:
        st.error("Column 'labels' missing in car parts.csv")
        st.stop()

    df["labels"] = df["labels"].astype(str).str.strip()
    df = df[df["labels"].notna()]
    df = df[df["labels"] != ""]

    classes = sorted(df["labels"].unique().tolist())

if len(classes) == 0:
    st.error("No valid classes found.")
    st.stop()

class_to_idx = {cls: i for i, cls in enumerate(classes)}
idx_to_class = {i: cls for cls, i in class_to_idx.items()}

# ==========================================
# Load Model (Developer Safe Version)
# ==========================================
@st.cache_resource
def load_model():

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # Try strict loading first
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        # If mismatch → load only matching layers
        filtered_state = {
            k: v for k, v in state_dict.items()
            if k in model.state_dict() and
            v.shape == model.state_dict()[k].shape
        }
        model.load_state_dict(filtered_state, strict=False)

    model.eval()
    return model

model = load_model()

# ==========================================
# Image Transform
# ==========================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==========================================
# Load Shops CSV Safely
# ==========================================
shops_df = pd.read_csv(SHOPS_CSV)

required_cols = ["name", "latitude", "longitude"]
for col in required_cols:
    if col not in shops_df.columns:
        st.error(f"Missing column: {col}")
        st.stop()

shops_df["latitude"] = pd.to_numeric(shops_df["latitude"], errors="coerce")
shops_df["longitude"] = pd.to_numeric(shops_df["longitude"], errors="coerce")
shops_df = shops_df.dropna(subset=["latitude", "longitude"])

# ==========================================
# Streamlit UI
# ==========================================
st.title("🔍 Car Part Identifier & Nearby Shop Finder")

uploaded_file = st.file_uploader(
    "📸 Upload Car Part Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    predicted_part = idx_to_class.get(predicted.item(), "Unknown")
    confidence_score = confidence.item() * 100

    st.success(
        f"🧩 Predicted: **{predicted_part}** "
        f"(Confidence: {confidence_score:.2f}%)"
    )

    # ======================================
    # Location Feature
    # ======================================
    user_location = st.text_input("📍 Enter your location")

    if user_location:
        try:
            geolocator = Nominatim(user_agent="car-part-app")
            location = geolocator.geocode(user_location, timeout=10)

            if location:
                user_coords = (location.latitude, location.longitude)

                st.map(pd.DataFrame([{
                    "lat": location.latitude,
                    "lon": location.longitude
                }]))

                st.markdown("### 🛠 Nearby Shops (within 80 km)")

                nearby = []

                for _, shop in shops_df.iterrows():
                    try:
                        shop_coords = (
                            float(shop["latitude"]),
                            float(shop["longitude"])
                        )

                        distance = geodesic(
                            user_coords,
                            shop_coords
                        ).km

                        if distance <= 80:
                            nearby.append({
                                "name": shop["name"],
                                "distance": round(distance, 2)
                            })
                    except:
                        continue

                if nearby:
                    for shop in sorted(nearby, key=lambda x: x["distance"]):
                        st.markdown(
                            f"- **{shop['name']}** — {shop['distance']} km"
                        )
                else:
                    st.info("No shops found within 80 km.")
            else:
                st.error("Location not found.")

        except:
            st.error("Geolocation service unavailable.")
