#Source:https://github.com/opencv/opencv/tree/master/data/haarcascades
import cv2
import os

#load the haar cascade for smile detection
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
#create a directory to store images
os.makedirs('detected_smiles', exist_ok =True)
image_count = 1

#initialize webcam.
cap = cv2.VideoCapture(0)

#main loop
while True:
    #read the current frames form the cam
    ret, frame = cap.read()

    #convert to grayscale
    gray= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #perform smile detection using the Haar cascade
    smiles = smile_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    #draw boundingboxes around the detected faces and save images
    for (x,y,w,h) in smiles:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        #save the detected smile
        smile_image=frame[y:y+h,x:x+w]
        filename=f'detected_smiles/smiles_{image_count}.jpg'
        cv2.imwrite(filename, smile_image)
        print(f'saved smile image:{filename}')
        image_count+=1

    #display the smile with bounding boxes
    cv2.imshow('smile detection',frame)

    #check for the key press to exit
    if cv2.waitKey(1) == ord('q'):
        break

#clean up
cap.release()
cv2.destroyAllWindows()
