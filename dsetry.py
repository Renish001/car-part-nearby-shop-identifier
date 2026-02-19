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

# ===============================
# Base Paths
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PARTS_CSV = os.path.join(BASE_DIR, "car parts.csv")
SHOPS_CSV = os.path.join(BASE_DIR, "coimbatore_spare_parts_shops.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model.pth")

# ===============================
# Load Parts CSV Safely
# ===============================
if not os.path.exists(PARTS_CSV):
    st.error("car parts.csv not found")
    st.stop()

df = pd.read_csv(PARTS_CSV)

if "labels" not in df.columns:
    st.error("Column 'labels' missing in car parts.csv")
    st.stop()

# Clean labels
df["labels"] = df["labels"].astype(str).str.strip()
df = df[df["labels"] != ""]
df = df[df["labels"].notna()]

classes = sorted(df["labels"].unique().tolist())

if len(classes) == 0:
    st.error("No valid labels found in dataset.")
    st.stop()

class_to_idx = {cls: i for i, cls in enumerate(classes)}
idx_to_class = {i: cls for cls, i in class_to_idx.items()}

# ===============================
# Load Model (Cached Properly)
# ===============================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model

if not os.path.exists(MODEL_PATH):
    st.error("model.pth not found")
    st.stop()

model = load_model()

# ===============================
# Image Transform
# ===============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ===============================
# Load Shops CSV Safely
# ===============================
if not os.path.exists(SHOPS_CSV):
    st.error("Shop CSV not found")
    st.stop()

shops_df = pd.read_csv(SHOPS_CSV)

required_cols = ["name", "latitude", "longitude"]

for col in required_cols:
    if col not in shops_df.columns:
        st.error(f"Missing column: {col}")
        st.stop()

shops_df["latitude"] = pd.to_numeric(shops_df["latitude"], errors="coerce")
shops_df["longitude"] = pd.to_numeric(shops_df["longitude"], errors="coerce")
shops_df = shops_df.dropna(subset=["latitude", "longitude"])

# ===============================
# Streamlit UI
# ===============================
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

    # ===============================
    # Location Input
    # ===============================
    user_location = st.text_input("📍 Enter your location (e.g., Coimbatore)")

    if user_location:
        try:
            geolocator = Nominatim(user_agent="car-part-app")
            location = geolocator.geocode(user_location, timeout=10)

            if location:
                user_coords = (
                    float(location.latitude),
                    float(location.longitude)
                )

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

                    except Exception:
                        continue

                if nearby:
                    for shop in sorted(nearby, key=lambda x: x["distance"]):
                        st.markdown(
                            f"- **{shop['name']}** — {shop['distance']} km"
                        )
                else:
                    st.info("No shops found within 80 km.")

            else:
                st.error("Location not found. Try a more specific place.")

        except Exception:
            st.error("Geolocation service unavailable. Try again later.")
