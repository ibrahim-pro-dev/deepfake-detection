import streamlit as st
from PIL import Image
from detector import detector

st.set_page_config(
    page_title="AI Deepfake Detection",
    layout="centered"
)

st.title("AI Deepfake Detection")
st.write(
    "Upload a real or AI-generated image to get a risk score, or upload a "
    "short video clip. The system is probabilistic and never infallible."
)

tab_images, tab_videos = st.tabs(
    ["Image", "Video"]
)

# ---------------------------------------------------------------------------
# Image tab
# ---------------------------------------------------------------------------

with tab_images:
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file)

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

        label, confidence, risk, overlay = (
            detector.predict_with_heatmap(image)
        )

        st.subheader("Result")

        if risk >= 0.5:
            st.error(
                "❌ FAKE — AI-generated / Deepfaked image "
                "(not taken by any camera)"
            )
        elif label.lower() == "no face detected":
            st.warning(label)
        else:
            st.success(
                "✅ REAL PHOTO — taken by a real camera/phone"
            )

        st.metric(
            "Risk Score",
            f"{risk*100:.2f}%"
        )
        st.progress(risk)
        st.caption(
            "Risk is the probability (0-100%) that this image is "
            "AI-generated, based on the model's output."
        )

        if overlay is not None:
            col_heat, col_raw = st.columns(2)
            with col_heat:
                st.image(
                    overlay,
                    caption="Grad-CAM Saliency Overlay",
                    use_container_width=True
                )
            with col_raw:
                st.image(
                    image,
                    caption="Original",
                    use_container_width=True
                )

        st.metric(
            "Predicted Class",
            f"{label} ({confidence*100:.2f}%)"
        )

# ---------------------------------------------------------------------------
# Video tab
# ---------------------------------------------------------------------------

with tab_videos:
    video_file = st.file_uploader(
        "Choose a short video clip",
        type=["mp4", "mov", "avi"]
    )

    if video_file:
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(
            suffix=".mp4",
            delete=False
        ) as tmp:
            tmp.write(video_file.getvalue())
            tmp_path = tmp.name

        st.video(video_file)

        with st.spinner("Classifying video frames..."):
            result = detector.predict_video(
                tmp_path,
                sample_every=5
            )

        os.unlink(tmp_path)

        if result is None:
            st.error("Could not read the video file.")
        else:
            label, confidence, risk, overlay = result

            st.subheader("Result")

            if risk >= 0.5:
                st.error("❌ AI GENERATED VIDEO")
            else:
                st.success("✅ REAL VIDEO")

            st.metric(
                "Risk Score",
                f"{risk*100:.2f}%"
            )
            st.progress(risk)

            if overlay is not None:
                st.image(
                    overlay,
                    caption="Highest-Risk Frame (Grad-CAM Saliency)",
                    use_container_width=True
                )

            st.metric(
                "Predicted Class",
                f"{label} ({confidence*100:.2f}%)"
            )

st.markdown("---")
st.caption(
    "💡 This is a research/demo tool. It reports a probabilistic risk score, "
    "not a guaranteed verdict — it is intended as one layer of defence, not "
    "a standalone guarantee."
)
