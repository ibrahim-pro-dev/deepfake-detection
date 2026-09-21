<<<<<<< HEAD
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
import cv2
import numpy as np
from PIL import Image

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if face_detector.empty():
    print("Face detector not loaded")
else:
    print("Face detector loaded successfully")

def extract_face(image):

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
        return None

    x, y, w, h = faces[0]

    face = img[y:y+h, x:x+w]

    return Image.fromarray(face)

class DeepfakeDetector:
    def __init__(self):
        self.device = "cpu"

        model_name = "Smogy/SMOGY-Ai-images-detector"

        self.processor = AutoImageProcessor.from_pretrained(model_name)

        self.model = AutoModelForImageClassification.from_pretrained(
            model_name
        )

        self.model.to(self.device)
        self.model.eval()

    def predict(self, image):
        image = extract_face(image)
        if image is None:
           return "No Face Detected", 0.0
        inputs = self.processor(
            images=image,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        probabilities = torch.nn.functional.softmax(
            outputs.logits,
            dim=1
        )

        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )

        label = self.model.config.id2label[prediction.item()]

        print("Predicted label:", label)
        print(self.model.config.id2label)

        return label, confidence.item()


=======
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
import cv2
import numpy as np
from PIL import Image

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if face_detector.empty():
    print("Face detector not loaded")
else:
    print("Face detector loaded successfully")

def extract_face(image):

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
        return None

    x, y, w, h = faces[0]

    face = img[y:y+h, x:x+w]

    return Image.fromarray(face)

class DeepfakeDetector:
    def __init__(self):
        self.device = "cpu"

        model_name = "Smogy/SMOGY-Ai-images-detector"

        self.processor = AutoImageProcessor.from_pretrained(model_name)

        self.model = AutoModelForImageClassification.from_pretrained(
            model_name
        )

        self.model.to(self.device)
        self.model.eval()

    def predict(self, image):
        image = extract_face(image)
        if image is None:
           return "No Face Detected", 0.0
        inputs = self.processor(
            images=image,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        probabilities = torch.nn.functional.softmax(
            outputs.logits,
            dim=1
        )

        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )

        label = self.model.config.id2label[prediction.item()]

        print("Predicted label:", label)
        print(self.model.config.id2label)

        return label, confidence.item()


>>>>>>> 73e7a979 (deepfake-detection)
detector = DeepfakeDetector()