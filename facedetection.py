Source: https://github.com/opencv/opencv/tree/master/data/haarcascades
import cv2
import os

#load the haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
#create a directory to store images
os.makedirs('detected_faces', exist_ok =True)
image_count = 1

#initialize webcam.
cap = cv2.VideoCapture(0)

#main loop
while True:
    #read the current frames form the cam
    ret, frame = cap.read()

    #convert to grayscale
    gray= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #perform face detection using the Haar cascade
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    #draw boundingboxes around the detected faces and save images
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        #save the detected face
        face_image=frame[y:y+h,x:x+w]
        filename=f'detected_faces/faces_{image_count}.jpg'
        cv2.imwrite(filename, face_image)
        print(f'saved face image:{filename}')
        image_count+=1

    #display the face with bounding boxes
    cv2.imshow('face detection',frame)

    #check for the key press to exit
    if cv2.waitKey(1) == ord('q'):
        break

#clean up
cap.release()
cv2.destroyAllWindows()
