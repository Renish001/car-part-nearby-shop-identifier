import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import pandas as pd
from geopy.distance import geodesic
import os

# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(page_title="Car Part Identifier", layout="wide")
st.title("🚗 Car Part Nearby Shop Identifier")

# -----------------------------------
# LOAD MODEL SAFELY
# -----------------------------------
@st.cache_resource
def load_model():

    model_path = "car_model.pth"

    if not os.path.exists(model_path):
        st.error("❌ Model file 'car_model.pth' not found in project folder.")
        st.stop()

    checkpoint = torch.load(model_path, map_location="cpu")

    classes = checkpoint.get("classes", None)

    if classes is None:
        st.error("❌ Classes not found inside model checkpoint.")
        st.stop()

    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(classes))

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, classes


model, classes = load_model()

# -----------------------------------
# IMAGE TRANSFORM (MUST MATCH TRAINING)
# -----------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------------
# IMAGE UPLOAD
# -----------------------------------
uploaded_file = st.file_uploader("Upload Car Part Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_index = torch.argmax(probabilities, dim=1).item()

    predicted_class = classes[predicted_index]

    st.success(f"🔍 Predicted Car Part: **{predicted_class}**")

    # -----------------------------------
    # LOCATION INPUT
    # -----------------------------------
    st.subheader("📍 Enter Your Location")

    user_lat = st.number_input("Latitude", format="%.6f")
    user_lon = st.number_input("Longitude", format="%.6f")

    # -----------------------------------
    # LOAD SHOP DATA SAFELY
    # -----------------------------------
    shop_file = "shops.csv"

    if not os.path.exists(shop_file):
        st.error("❌ shops.csv file not found.")
        st.stop()

    df = pd.read_csv(shop_file)

    required_columns = ["shop_name", "address", "latitude", "longitude", "part_name"]

    for col in required_columns:
        if col not in df.columns:
            st.error(f"❌ Missing column in shops.csv: {col}")
            st.stop()

    if st.button("Find Nearby Shops"):

        if user_lat == 0 or user_lon == 0:
            st.error("Please enter valid latitude and longitude.")
        else:
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
                st.warning("No shops found selling this part near your location.")