import cv2
import requests
from plyer import notification

# Load the pre-trained face recognizer and cascade classifier
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

# Define font and user names
font = cv2.FONT_HERSHEY_SIMPLEX
names = ['', 'Varun']

# Open the camera
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # set video width
cam.set(4, 480)  # set video height
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

# Flag to control the loop
flag = True

def send_access_signal():
    # Send HTTP request to Arduino server's /access_signal route
    url = 'http://192.168.4.1/access_signal'  # Replace with your Arduino server's IP address
    try:
        response = requests.get(url)
        if response.status_code == 200:
            print("Access signal sent successfully")
        else:
            print("Failed to send access signal")
    except Exception as e:
        print("Error sending access signal:", e)

def face_recognition():
    global flag
    while flag:
        ret, img = cam.read()  # Read a frame from the camera
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(int(minW), int(minH)))

        # Iterate through detected faces
        for (x, y, w, h) in faces:
            # Recognize the face
            id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

            # Check if the confidence level is below a certain threshold
            if confidence < 100:
                id = names[id]
                confidence = "  {0}%".format(round(100 - confidence) + 30)

                print("Welcome", id)
                send_access_signal()  # Send access signal to Arduino server
                flag = False
            else:
                notification.notify(
                    title="Unidentified person",
                    message="Someone unidentified is at your door",
                    app_icon="icon.ico",
                    timeout=5
                )

            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
            cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (2, 255, 0), 1)

        cv2.imshow('camera', img)
        k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
        if k == 27:
            break

    print("\n[INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()

# Start the face recognition process
face_recognition()
