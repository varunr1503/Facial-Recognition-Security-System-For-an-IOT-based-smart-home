import cv2
from plyer import notification
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')   
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
font = cv2.FONT_HERSHEY_SIMPLEX
id = 2 
names = ['','Varun']  
cam = cv2.VideoCapture(0)
cam.set(3, 640) 
cam.set(4, 480) 
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)
flag = True
while flag:
    ret, img =cam.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,0), 2)
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100-confidence)+30)
        else:
            id = "Varun"
            confidence = "  {0}%".format(round(100-confidence)+30)
            if __name__ == "__main__":
                notification.notify(
                    title= "Unidentified person",
                    message="Someone unidentified is at your door",
                    app_icon= "icon.ico",
                    timeout=5   )
        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (2,255,0), 1)  
    cv2.imshow('camera',img) 
    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
