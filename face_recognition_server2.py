import cv2
import requests
from plyer import notification
import os
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from facenet_pytorch import InceptionResnetV1
import joblib

# Load FaceNet model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Define the path to your dataset directory
dataset_path = 'dataset'

# Load the pre-trained classifier
classifier = joblib.load('trainer/classifier.pkl')

# Open the camera
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # set video width
cam.set(4, 480)  # set video height
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

# Flag to control the loop
flag = True

def send_access_signal(user_id):
    # Send HTTP request to Arduino server's /access_signal route
    url = 'http://192.168.4.1/access_signal'  # Replace with your Arduino server's IP address
    try:
        response = requests.get(url)
        if response.status_code == 200:
            print("Access signal sent successfully for user:", user_id)
        else:
            print("Failed to send access signal")
    except Exception as e:
        print("Error sending access signal:", e)

def face_recognition():
    global flag
    while flag:
        ret, img = cam.read()  # Read a frame from the camera
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
        faces = detect_faces(gray)

        # Iterate through detected faces
        for (x, y, w, h) in faces:
            # Generate embedding using FaceNet
            face_img = img[y:y + h, x:x + w]
            embedding = get_embedding(face_img)
            
            # Recognize the face
            user_id = recognize_face(embedding)
            
            if user_id is not None:
                name = get_user_name(user_id)
                confidence = "High"  # You can customize this based on the threshold confidence
                
                print("Welcome", name)
                # send_access_signal(user_id)  # Send access signal to Arduino server
                # flag = False
            else:
                notification.notify(
                    title="Unidentified person",
                    message="Someone unidentified is at your door",
                    app_icon="icon.ico",
                    timeout=5
                )

            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
            cv2.putText(img, name, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(img, str(confidence), (x + 5, y + h - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (2, 255, 0), 1)

        cv2.imshow('camera', img)
        k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
        if k == 27:
            break

    print("\n[INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()

def detect_faces(gray):
    cascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(int(minW), int(minH)))
    return faces

def get_embedding(face_img):
    # Preprocess face image for FaceNet
    img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
    img = transforms.Resize((160, 160))(img)
    img = transforms.ToTensor()(img)
    img = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(img)
    img = img.unsqueeze(0).to(device)
    
    # Generate embedding
    with torch.no_grad():
        embedding = model(img).cpu().numpy().flatten()
    
    return embedding

def recognize_face(embedding):
    # Use the pre-trained classifier to predict the user ID
    user_id = classifier.predict([embedding])[0]
    return user_id

def get_user_name(user_id):
    # Replace this with your logic to retrieve user names from your dataset
    return "User " + "Varun"

# Start the face recognition process
face_recognition()
