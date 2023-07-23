#This code will enable you to play a background audio after detecting the smile


import cv2
import os
from pydub import AudioSegment
from pydub.playback import play

# Load the haar cascade for smile detection
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
# Create a directory to store images
os.makedirs('detected_smiles', exist_ok=True)
image_count = 1

# Initialize webcam.
cap = cv2.VideoCapture(0)

# Load the audio file
audio_file_path = "E:\AI 2\cacscade\sample1.mp3"  # Replace with your audio file path
audio = AudioSegment.from_file(audio_file_path)

# Flag to track if audio is playing or not
audio_playing = False

# Main loop
while True:
    # Read the current frames from the cam
    ret, frame = cap.read()

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform smile detection using the Haar cascade
    smiles = smile_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Draw bounding boxes around the detected faces and save images
    for (x, y, w, h) in smiles:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Save the detected smile
        smile_image = frame[y:y + h, x:x + w]
        filename = f'detected_smiles/smiles_{image_count}.jpg'
        cv2.imwrite(filename, smile_image)
        print(f'saved smile image: {filename}')
        image_count += 1

        # Play audio when a smile is detected
        if not audio_playing:
            play(audio)
            audio_playing = True

    # Check if all smiles are gone, reset the audio_playing flag
    if len(smiles) == 0:
        audio_playing = False

    # Display the smile with bounding boxes
    cv2.imshow('smile detection', frame)

    # Check for the key press to exit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
