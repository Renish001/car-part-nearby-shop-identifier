import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import pandas as pd
from geopy.distance import geodesic

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Car Part Identifier", layout="wide")
st.title("🚗 Car Part Nearby Shop Identifier")

# -------------------------------
# LOAD MODEL (CACHED)
# -------------------------------
@st.cache_resource
def load_model():
    checkpoint = torch.load("car_model.pth", map_location=torch.device("cpu"))

    classes = checkpoint["classes"]

    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(classes))

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, classes

model, classes = load_model()

# -------------------------------
# IMAGE TRANSFORM (MUST MATCH TRAINING)
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------------------------------
# IMAGE UPLOAD
# -------------------------------
uploaded_file = st.file_uploader("Upload Car Part Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_tensor = transform(image)
    img_tensor = img_tensor.unsqueeze(0)

    # Prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_index = torch.argmax(probabilities, dim=1).item()

    predicted_class = classes[predicted_index]

    st.success(f"🔍 Predicted Car Part: **{predicted_class}**")

    # -------------------------------
    # USER LOCATION INPUT
    # -------------------------------
    st.subheader("📍 Enter Your Location")

    user_lat = st.number_input("Latitude", format="%.6f")
    user_lon = st.number_input("Longitude", format="%.6f")

    # -------------------------------
    # LOAD SHOP DATA
    # -------------------------------
    df = pd.read_csv("shops.csv")  
    # shops.csv must contain:
    # shop_name, address, latitude, longitude, part_name

    if st.button("Find Nearby Shops"):

        if user_lat != 0 and user_lon != 0:

            user_coords = (user_lat, user_lon)

            filtered_shops = df[df["part_name"] == predicted_class]

            results = []

            for _, row in filtered_shops.iterrows():
                try:
                    shop_coords = (float(row["latitude"]), float(row["longitude"]))
                    distance_km = geodesic(user_coords, shop_coords).km

                    results.append({
                        "Shop Name": row["shop_name"],
                        "Address": row["address"],
                        "Distance (KM)": round(distance_km, 2)
                    })

                except:
                    continue

            if results:
                results_df = pd.DataFrame(results)
                results_df = results_df.sort_values(by="Distance (KM)")
                st.subheader("🏪 Nearby Shops")
                st.dataframe(results_df)
            else:
                st.warning("No shops found selling this part.")

        else:
            st.error("Please enter valid latitude and longitude.")