import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import pandas as pd
from PIL import Image
import os
from geopy.distance import geodesic
from geopy.geocoders import Nominatim

# ==============================
# Base Directory
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PARTS_CSV = os.path.join(BASE_DIR, "car parts.csv")
SHOPS_CSV = os.path.join(BASE_DIR, "coimbatore_spare_parts_shops.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model.pth")

# ==============================
# Load Parts CSV
# ==============================
if not os.path.exists(PARTS_CSV):
    st.error(f"Parts dataset not found at {PARTS_CSV}")
    st.stop()

df = pd.read_csv(PARTS_CSV)

if "labels" not in df.columns:
    st.error("Column 'labels' not found in car parts.csv")
    st.stop()

class_to_idx = {name: idx for idx, name in enumerate(df["labels"].unique())}
idx_to_class = {v: k for k, v in class_to_idx.items()}

# ==============================
# Load Model Safely
# ==============================
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at {MODEL_PATH}")
    st.stop()

model = models.resnet18(weights=None)

checkpoint = torch.load(MODEL_PATH, map_location=torch.device("cpu"))

# Handle different saving formats
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
else:
    state_dict = checkpoint

# Remove FC layer weights (to prevent size mismatch)
filtered_state_dict = {
    k: v for k, v in state_dict.items()
    if not k.startswith("fc.")
}

model.load_state_dict(filtered_state_dict, strict=False)

# Recreate final layer properly
model.fc = nn.Linear(model.fc.in_features, len(class_to_idx))

model.eval()

# ==============================
# Image Transform
# ==============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==============================
# Load Shops CSV Safely
# ==============================
if not os.path.exists(SHOPS_CSV):
    st.error(f"Shop dataset not found at {SHOPS_CSV}")
    st.stop()

shops_df = pd.read_csv(SHOPS_CSV)

required_columns = ["name", "latitude", "longitude"]

for col in required_columns:
    if col not in shops_df.columns:
        st.error(f"Missing column '{col}' in shop dataset")
        st.stop()

# Convert coordinates safely
shops_df["latitude"] = pd.to_numeric(shops_df["latitude"], errors="coerce")
shops_df["longitude"] = pd.to_numeric(shops_df["longitude"], errors="coerce")

shops_df = shops_df.dropna(subset=["latitude", "longitude"])

# ==============================
# Streamlit UI
# ==============================
st.title("🔍 Car Part Identifier & Nearby Shop Finder")

uploaded_file = st.file_uploader(
    "📸 Upload a car part image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        predicted_part = idx_to_class.get(predicted.item(), "Unknown")

    st.success(f"🧩 Predicted Car Part: **{predicted_part}**")

    # ==============================
    # Location Input
    # ==============================
    user_location = st.text_input(
        "📍 Enter your location (e.g., Coimbatore)"
    )

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

                st.markdown("### 🛠️ Nearby Repair Shops (within 80 km)")

                nearby_shops = []

                for _, shop in shops_df.iterrows():
                    try:
                        shop_coords = (
                            float(shop["latitude"]),
                            float(shop["longitude"])
                        )

                        distance_km = geodesic(
                            user_coords,
                            shop_coords
                        ).km

                        if distance_km <= 80:
                            nearby_shops.append({
                                "name": shop["name"],
                                "distance": round(distance_km, 2)
                            })

                    except Exception:
                        continue

                if nearby_shops:
                    for shop in sorted(
                        nearby_shops,
                        key=lambda x: x["distance"]
                    ):
                        st.markdown(
                            f"- **{shop['name']}** — {shop['distance']} km"
                        )
                else:
                    st.info("No nearby shops found within 80 km.")

            else:
                st.error("Couldn't find your location. Try a more specific name.")

        except Exception:
            st.error("Location service unavailable. Please try again later.")
