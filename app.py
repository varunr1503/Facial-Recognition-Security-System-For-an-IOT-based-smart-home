from flask import Flask, render_template, jsonify,request
import cv2
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
import joblib
import os   

app = Flask(__name__)

# Load FaceNet model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Load the pre-trained classifier
classifier = joblib.load('trainer/classifier.pkl')

def get_embeddings_and_labels(dataset_path):
    image_paths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, f))]
    embeddings = []
    labels = []

    for image_path in image_paths:
        img = Image.open(image_path).convert('RGB')
        img = transforms.Resize((160, 160))(img)
        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(img)
        img = img.unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = model(img).cpu().numpy().flatten()

        embeddings.append(embedding)
        
        # Extract user ID from the image file name
        user_id = int(os.path.basename(image_path).split(".")[1])
        labels.append(user_id)

    return np.array(embeddings), np.array(labels)

def train_classifier(embeddings, labels):
    from sklearn.svm import SVC
    classifier = SVC(kernel='linear')
    classifier.fit(embeddings, labels)
    return classifier

def save_classifier(classifier):
    import joblib
    joblib.dump(classifier, 'trainer/classifier.pkl')


# Open the camera
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # set video width
cam.set(4, 480)  # set video height
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

# Flag to track access status
access_granted = False

@app.route('/')
def index():
    if access_granted:
        return render_template('secure_page.html')
    else:
        return "Access Denied"
    
@app.route('/iot_devices')
def iot_devices():
    if access_granted:
        return render_template('iot_devices.html')
    else:
        return "Access Denied"

dataset_path = 'dataset'

def train():
    # Get embeddings and labels from the dataset
    embeddings, labels = get_embeddings_and_labels(dataset_path)
    # Train the classifier
    classifier = train_classifier(embeddings, labels)
    # Save the trained classifier
    save_classifier(classifier)
    

@app.route('/register_profile', methods=['GET', 'POST'])
def register_profile():
    if request.method == 'GET':
        return render_template('register_profile.html')
    elif request.method == 'POST':
        # Get user ID from the form submission
        user_id = request.form['user_id']
        
        # Use OpenCV to capture images of the user's face
        capture_images(int(user_id))
        
        # After capturing images, trigger training process
        train()
        
        return render_template('registration_complete.html')

def capture_images(user_id):
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)
    cam.set(4, 480) 
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    print("\n [INFO] Initializing face capture. Look at the camera and wait ...")

    count = 0

    while(True):
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
            count += 1

            cv2.imwrite(f"{dataset_path}/User.{user_id}_{count}.jpg", gray[y:y+h,x:x+w])
            cv2.imshow('image', img)

        k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
        if k == 27:
            break
        elif count >= 30:
             break

    print("\n [INFO] Exiting Program and cleaning up stuff")
    cam.release()
    cv2.destroyAllWindows()

def recognize_user():
    global access_granted
    # while True:
    #     ret, img = cam.read()  # Read a frame from the camera
    #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
    #     faces = detect_faces(gray)
    #     # Iterate through detected faces
    #     for (x, y, w, h) in faces:
    #         # Generate embedding using FaceNet
    #         face_img = img[y:y + h, x:x + w]
    #         embedding = get_embedding(face_img)
            
    #         # Recognize the face
    #         user_id = recognize_face(embedding)
            
    #         if user_id is not None:
    #             access_granted = True
    #             return
    #         else:
    #             return  
    access_granted = True
    return
        

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

if __name__ == '__main__':
    # Simulate user recognition
    recognize_user()
    app.run(debug=True)
