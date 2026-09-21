import torch
import torch.nn.functional as F
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
)
import cv2
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Face detection (OpenCV Haar cascades)
# ---------------------------------------------------------------------------

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if face_detector.empty():
    print("Warning: face detector not loaded")
else:
    print("Face detector loaded successfully")


def extract_face(image):
    """Crop the largest detected face.

    Returns (face_image, box) or (None, None) if no face was found.
    box is the (x, y, w, h) rectangle in the original image coordinates.
    """

    img = np.array(image.convert("RGB"))

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2GRAY
    )

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    if len(faces) == 0:
        return None, None

    x, y, w, h = faces[0]

    face = img[y:y+h, x:x+w]

    return Image.fromarray(face), (x, y, w, h)


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

AI_TOKENS = ("ai", "artificial", "fake", "generated", "synthetic")


class DeepfakeDetector:
    def __init__(self):
        self.device = "cpu"

        model_name = "dima806/deepfake_vs_real_image_detection"

        self.processor = AutoImageProcessor.from_pretrained(model_name)

        self.model = AutoModelForImageClassification.from_pretrained(
            model_name
        )

        self.model.to(self.device)
        self.model.eval()

    def _score(self, face):
        """Run the model with gradients enabled.

        Returns (label, confidence, risk, outputs, inputs).
        """

        inputs = self.processor(
            images=face,
            return_tensors="pt"
        )
        inputs["pixel_values"].requires_grad_(True)

        outputs = self.model(**inputs)

        probabilities = F.softmax(
            outputs.logits,
            dim=1
        )

        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )

        label = self.model.config.id2label[prediction.item()]

        is_ai = any(
            token in label.lower()
            for token in AI_TOKENS
        )

        risk = confidence.item() if is_ai else 1.0 - confidence.item()

        print("Predicted label:", label)
        print(self.model.config.id2label)

        return label, confidence.item(), risk, outputs, inputs

    def predict(self, image):
        """Classify a single image.

        Returns (label, confidence, risk).
        """

        face, _ = extract_face(image)

        if face is None:
            return "No Face Detected", 0.0, 0.0

        with torch.no_grad():
            label, confidence, risk, _, _ = self._score(face)

        return label, confidence, risk

    def predict_with_heatmap(self, image):
        """Classify and build a gradient-saliency overlay (Grad-CAM style).

        Returns (label, confidence, risk, overlay_image).
        overlay_image is the original image with a jet heatmap blended over
        the face (None if no face was found).
        """

        img = np.array(image.convert("RGB"))
        img = img.copy()

        face, box = extract_face(image)

        if face is None:
            return "No Face Detected", 0.0, 0.0, None

        x, y, w, h = box

        label, confidence, risk, outputs, inputs = self._score(face)

        # ---- gradient saliency on the winning class ----
        self.model.zero_grad()

        probabilities = F.softmax(
            outputs.logits,
            dim=1
        )
        _, prediction = torch.max(
            probabilities,
            dim=1
        )

        target = outputs.logits[0, prediction.item()]
        target.backward()

        grad = inputs["pixel_values"].grad[0]

        saliency = grad.abs().mean(dim=0).cpu().numpy()

        saliency = cv2.normalize(
            saliency,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)

        heat = cv2.applyColorMap(
            saliency,
            cv2.COLORMAP_JET
        )
        heat = cv2.resize(
            heat,
            (w, h)
        )

        face_region = img[y:y+h, x:x+w]

        overlay_region = cv2.addWeighted(
            heat,
            0.6,
            face_region,
            0.4,
            0
        )

        img[y:y+h, x:x+w] = overlay_region

        return label, confidence, risk, Image.fromarray(img)

    def predict_video(self, video_path, sample_every=5):
        """Classify a short video by sampling frames.

        Returns (label, confidence, risk, overlay_image) of the worst
        (highest-risk) sampled frame, or None if the video could not be read.
        """

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            return None

        results = []
        frame_index = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index % sample_every == 0:
                rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                result = self.predict_with_heatmap(
                    Image.fromarray(rgb)
                )

                if result is not None:
                    results.append(result)

            frame_index += 1

        cap.release()

        if not results:
            return None

        # Aggregate risk across ALL sampled frames. A single ambiguous frame
        # can no longer flip the whole clip: we report the mean verdict and
        # the frame closest to it as the illustrative heatmap.
        mean_risk = sum(
            r[2] for r in results
        ) / len(results)

        closest = min(
            results,
            key=lambda r: abs(r[2] - mean_risk)
        )

        return closest[0], closest[1], mean_risk, closest[3]


# ---------------------------------------------------------------------------
# Shared instance used by the app
# ---------------------------------------------------------------------------

detector = DeepfakeDetector()
