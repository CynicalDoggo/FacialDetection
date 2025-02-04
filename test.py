import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
import numpy as np
import cv2
import os
import time

# --- Model Loading ---
def load_ssd_model():
    """Load the SSD MobileNetV2 model for face detection from the current directory."""
    model_dir = os.getcwd()  # Get current working directory
    try:
        print("Loading SSD MobileNetV2 model from the current directory...")
        ssd_model = tf.saved_model.load(model_dir)
        print("SSD MobileNetV2 model loaded successfully!")
        return ssd_model
    except Exception as e:
        print(f"Error loading SSD model: {e}")
        exit()


def load_embedding_model():
    """Load the MobileNetV2 model for embedding generation."""
    print("Loading MobileNetV2 model...")
    base_model = keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    x = base_model.output
    x = keras.layers.GlobalAveragePooling2D()(x)
    embedding_model = keras.models.Model(inputs=base_model.input, outputs=x)
    print("MobileNetV2 model loaded successfully!")
    return embedding_model

def load_mask_detection_model():
    """Load the mask detection model."""
    print("Loading mask detection model...")
    model = keras.models.load_model('mode_maskDetect.h5')  # Replace with your mask detection model path
    print("Mask detection model loaded successfully!")
    return model

ssd_model = load_ssd_model()
embedding_model = load_embedding_model()
mask_model = load_mask_detection_model()

# --- Directory Setup ---
embedding_dir = "embeddings/"
os.makedirs(embedding_dir, exist_ok=True)

# --- Helper Functions ---
def preprocess_face(image):
    """Preprocess and prepare the face image for embedding and mask detection."""
    face = cv2.resize(image, (224, 224))
    face = face / 255.0  # Normalize the image
    return np.expand_dims(face, axis=0)

def detect_faces(img):
    """Detect faces using SSD MobileNetV2."""
    input_tensor = tf.convert_to_tensor([cv2.resize(img, (320, 320))], dtype=tf.uint8)
    detections = ssd_model(input_tensor)
    boxes = detections['detection_boxes'][0].numpy()
    scores = detections['detection_scores'][0].numpy()
    return boxes, scores

def save_embedding(name, embedding):
    """Save normalized embedding."""
    embedding = embedding / np.linalg.norm(embedding)
    np.save(os.path.join(embedding_dir, f"{name}.npy"), embedding)
    print(f"Embedding for {name} saved successfully!")

# --- Main Script ---
print("Starting Capture Phase...")

# Prompt for the name at the start
name = input("Enter your name for saving the embedding: ").strip()
if not name:
    print("Name cannot be empty. Exiting...")
    exit()

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

face_detected_time = None

while True:
    success, imgOriginal = cap.read()
    if not success:
        print("Failed to capture frame. Exiting...")
        break

    # Detect faces
    boxes, scores = detect_faces(imgOriginal)
    height, width, _ = imgOriginal.shape

    for i, score in enumerate(scores):
        if score < 0.5:  # Confidence threshold
            continue

        ymin, xmin, ymax, xmax = boxes[i]
        x, y, w, h = int(xmin * width), int(ymin * height), int((xmax - xmin) * width), int((ymax - ymin) * height)
        face = imgOriginal[y:y + h, x:x + w]

        # Mask detection
        preprocessed_face = preprocess_face(face)
        mask_prediction = mask_model.predict(preprocessed_face)[0]
        mask_label = "Mask" if mask_prediction[0] > 0.5 else "No Mask"

        # Draw rectangle and label
        color = (0, 255, 0) if mask_label == "No Mask" else (0, 0, 255)
        cv2.rectangle(imgOriginal, (x, y), (x + w, y + h), color, 2)
        cv2.putText(imgOriginal, mask_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        if mask_label == "No Mask":
            if face_detected_time is None:
                face_detected_time = time.time()

            if time.time() - face_detected_time >= 5:
                embedding = embedding_model.predict(preprocessed_face)[0]
                if embedding.shape == (1280,):
                    save_embedding(name, embedding)
                    print("Embedding saved successfully!")
                else:
                    print("Embedding shape mismatch.")
                cap.release()
                cv2.destroyAllWindows()
                exit()
        else:
            face_detected_time = None

    # Display the frame
    cv2.imshow("Face Capture", imgOriginal)

    # Exit on 'q'
    if cv2.waitKey(1) == ord('q'):
        print("Exiting without capturing.")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
