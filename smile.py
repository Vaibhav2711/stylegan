import cv2
import numpy as np
import os 

def detect(frame):
    # detect faces within the greyscale version of the frame
    img = cv2.imread(frame)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    #eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')
    gray = cv2.cvtColor(np.float32(img), cv2.COLOR_BGR2GRAY)
    gray = np.array(gray, dtype='uint8')
    faces = face_cascade.detectMultiScale(gray, 1.3, 20)
    #print(faces)
    pose = ""
    smile = ""
    if len(faces):
        pose = "Straight"
    else:
        print = "Side pose"
    num_smiles = 0
    for (x, y, w, h) in faces:
        
        #cv2.rectangle(frame, (x, y), ((x + w), (y + h)), (255, 0, 0), 2)

        # Calculate the "region of interest", ie the are of the frame
        # containing the face
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Within the grayscale ROI, look for smiles 
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)

        # If we find smiles then increment our counter
        if len(smiles):
            smile = "Smile"
        else:
            smile = "No smile"
    return smile,pose




