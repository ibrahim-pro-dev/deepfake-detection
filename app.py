<<<<<<< HEAD
import streamlit as st
from PIL import Image
from detector import detector

st.title("AI Deepfake Detection")

st.write("Upload an image to check whether it is Real or Deepfake.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    result, confidence = detector.predict(image)

    st.subheader("Result")

    if result.lower() == "human":
       st.success("✅ REAL IMAGE")
       st.metric("Confidence", f"{confidence*100:.2f}%")
    elif result.lower() == "artificial":
       st.error("❌ AI GENERATED IMAGE")
       st.metric("Confidence", f"{confidence*100:.2f}%")
    else:
       st.warning(result)
=======
import streamlit as st
from PIL import Image
from detector import detector

st.title("AI Deepfake Detection")

st.write("Upload an image to check whether it is Real or Deepfake.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    result, confidence = detector.predict(image)

    st.subheader("Result")

    if result.lower() == "human":
       st.success("✅ REAL IMAGE")
       st.metric("Confidence", f"{confidence*100:.2f}%")
    elif result.lower() == "artificial":
       st.error("❌ AI GENERATED IMAGE")
       st.metric("Confidence", f"{confidence*100:.2f}%")
    else:
       st.warning(result)
>>>>>>> 73e7a979 (deepfake-detection)
       st.metric("Confidence", f"{confidence*100:.2f}%")