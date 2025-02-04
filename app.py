#FACIAL DETECTION APP
#==========================================================================================
from flask import Flask, request, jsonify
from flask_cors import CORS
from supabase import create_client, client
from dotenv import load_dotenv
from tensorflow import keras
import tensorflow as tf
import numpy as np
import base64
import cv2
import os

#Supabase Configuration
load_dotenv(dotenv_path="C:/Users/Bryant Tan/OneDrive/Desktop/FacialDetection/.env")
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
supabase = create_client(url, key)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:5173", "http://127.0.0.1:5173"]}}, supports_credentials=True)

# --- Model Loading ---
#Facial Detection Model
def load_ssd_model():
    try:
        model_dir = os.getcwd() 
        ssd_model = tf.saved_model.load(model_dir)
        return ssd_model
    except Exception as e:
        print(f"Error loading SSD model: {e}")
        exit()

#Embedding generator
def load_embedding_model():
    try:
        base_model = keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
        x = base_model.output
        x = keras.layers.GlobalAveragePooling2D()(x)
        embedding_model = keras.models.Model(inputs=base_model.input, outputs=x)
        return embedding_model
    except Exception as e:
        print(f"Error loading MobileNetV2 Embedding: {e}")

#Mask Detection model
def load_mask_detection_model():
    try:
        model = keras.models.load_model('mode_maskDetect.h5')  # Replace with your mask detection model path
        return model
    except Exception as e:
        print(f"Error while loading Mask Detection model: {e}")

ssd_model = load_ssd_model()
embedding_model = load_embedding_model()
mask_model = load_mask_detection_model()

# --- Helper Functions ---
    #Preprocess and prepare the face image for embedding and mask detection.
def preprocess_face(img):
    face = cv2.resize(img, (224, 224))
    face = face / 255.0  # Normalize the image
    return np.expand_dims(face, axis=0)

def detect_faces(img):
    resized = cv2.resize(img, (320, 320))
    input_tensor = tf.convert_to_tensor([resized], dtype=tf.uint8)
    detections = ssd_model(input_tensor)
    boxes = detections['detection_boxes'][0].numpy()
    scores = detections['detection_scores'][0].numpy()
    return boxes, scores

def generate_embedding(img):
    preprocessed_face = preprocess_face(img)
    embedding = embedding_model.predict(preprocessed_face)[0]
    return embedding / np.linalg.norm(embedding)  # Normalize embedding

def detect_mask(img):
    preprocessed_face = preprocess_face(img)
    mask_prediction = mask_model.predict(preprocessed_face)[0]
    return "Mask" if mask_prediction[0] > 0.5 else "No Mask"

# --- API Endpoints ---
@app.route('/detect_faces_fxn', methods=['POST'])
def detect_faces_fxn():
    try:
        data = request.json
        image_data = data["image"]  # Base64 encoded image
        
        # Decode Base64 image to OpenCV format
        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Detect faces
        boxes, scores = detect_faces(img)  # Use your existing `detect_faces` function
        height, width, _ = img.shape
        
        detections = []
        for i, score in enumerate(scores):
            if score < 0.5:  # Confidence threshold for face detection
                continue
            
            #Insid the /detect_faces_fxn endpoint loop:
            ymin, xmin, ymax, xmax = boxes[i]

            # Calculate coordinates with tighter box
            x = int(xmin * width)
            y = int(ymin * height)
            w = int((xmax - xmin) * width * 0.8)  # Reduce width by 20%
            h = int((ymax - ymin) * height * 0.8)  # Reduce height by 20%

            # Center the bounding box around the detected face
            x = x + int(w * 0.1)  # Adjust x to center
            y = y + int(h * 0.1)  # Adjust y to center

            face = img[y:y + h, x:x + w]

            # Mask detection
            preprocessed_face = preprocess_face(face)
            mask_prediction = mask_model.predict(preprocessed_face)[0]
            mask_label = "Mask" if mask_prediction[0] > 0.8 else "No Mask"  # Increased threshold

            detections.append({
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "label": mask_label,
                "confidence": float(mask_prediction[0])  # Optional: Include confidence
            })

            print(f"Mask confidence: {mask_prediction[0]}")

        return jsonify({"detections": detections})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Failed to process image"}), 500
@app.route('/save_embedding', methods=['POST'])
def save_embedding():
    try:
        data = request.get_json()  # Use get_json() for better parsing
        if not data or 'image' not in data or 'user_id' not in data or "facialRecognitionOptIn" not in data:
            return jsonify({"status": "error", "message": "Missing image or user_id"}), 400
        
        facialOptIn = data.get('facialRecognitionOptIn', True)
        image_data = data['image']
        user_id = data['user_id']

        # Validate user_id
        if not user_id:
            return jsonify({"status": "error", "message": "User ID is required"}), 400

        # Decode image
        try:
            image_bytes = base64.b64decode(image_data)
            image_np = np.frombuffer(image_bytes, dtype=np.uint8)
            img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
            if img is None:
                return jsonify({"status": "error", "message": "Invalid image data"}), 400
        except Exception as e:
            return jsonify({"status": "error", "message": "Failed to decode image"}), 400

        height, width = img.shape[:2]

        # Detect faces
        try:
            boxes, scores = detect_faces(img)
        except Exception as e:
            return jsonify({"status": "error", "message": "Face detection failed"}), 500

        for i, score in enumerate(scores):
            if score < 0.5:
                continue

            ymin, xmin, ymax, xmax = boxes[i]
            x = int(xmin * width)
            y = int(ymin * height)
            w = int((xmax - xmin) * width)
            h = int((ymax - ymin) * height)

            # Validate face region
            if w <= 0 or h <= 0 or x + w > width or y + h > height:
                continue

            face = img[y:y + h, x:x + w]
            if face.size == 0:
                continue

            # Mask detection
            try:
                mask_label = detect_mask(face)
                if mask_label == "Mask":
                    return jsonify({"status": "error", "message": "Mask detected"}), 400
            except Exception as e:
                return jsonify({"status": "error", "message": "Mask detection failed"}), 500

            # Generate embedding
            try:
                embedding = generate_embedding(face)
                embedding_list = embedding.tolist()
            except Exception as e:
                return jsonify({"status": "error", "message": "Embedding generation failed"}), 500

            # Update Supabase
            try:
                response = supabase.table('guest').update({'facial_data': embedding_list}).eq('user_id', user_id).execute()
                response = supabase.table('guest').update({'facialid_consent': facialOptIn}).eq('user_id', user_id).execute()
            except Exception as e:
                return jsonify({"status": "error", "message": f"Database update failed: {str(e)}"}), 500

            return jsonify({"status": "success", "message": "Embedding saved!"})

        return jsonify({"status": "error", "message": "No face detected"}), 400

    except Exception as e:
        print(f"Error in /save_embedding: {str(e)}")
        return jsonify({"status": "error", "message": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

