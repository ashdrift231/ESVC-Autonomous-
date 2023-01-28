import cv2
from tensorflow.keras.preprocessing.image import img_to_array
import os
import numpy as np
from tensorflow.keras.models import model_from_json


class FaceTonDetector:
    def __init__(self):
        self.texture = 0 #initiating flag variables
        self.face_cascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml") #loading haar feature for face 
        json_file = open('antispoofing_models/CS_on_IS_texture_based_FLD.json','r') 
        #loading the model
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        self.model.load_weights('antispoofing_models/CS_on_IS_TB_FLD_model_93-0.978947.h5')
        print("Model loaded")

    def detect_tone(self):
        # specify the path of the video file and create a VideoCapture object
        cap = cv2.VideoCapture("/home/ashdrift/Desktop/detect liveness/output.mp4")

        # check if the video file is opened successfully
        if not cap.isOpened():
            print("Error opening video file")

        while True: 
            _, frame = cap.read()    
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #converting to gray scale
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5) #detecting faces
            for (x,y,w,h) in faces:
                face = frame[y-5:y+h+5, x-5:x+w+5]
                resized_face = cv2.resize(face, (160, 160)) 
                resized_face = resized_face.astype("float") / 255.0
                resized_face = img_to_array(resized_face)
                resized_face = np.expand_dims(resized_face, axis=0)
                preds = self.model.predict(resized_face)[0] #making predictions
                if preds < 0.5: 
                    self.texture = 1 #setting flag
            cv2.imshow("Video", frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            else:
                break

    def FlagReturn(self): #returning flag
        return self.texture
        


'''while True:
        # Detect faces in the input image
        faces = face_cascade.detectMultiScale(ImgG,1.3,5)
        for (x,y,w,h) in faces:  
            face = ImgG[y-5:y+h+5,x-5:x+w+5]
            resized_face = cv2.resize(face,(160,160))
            resized_face = resized_face.astype("float") / 255.0
            resized_face = img_to_array(resized_face)
            resized_face = np.expand_dims(resized_face, axis=0)
            # model to determine if the face is "real" or "fake"
            preds = model.predict(resized_face)[0]
            if preds < 0.5: 
                texture = 1
                return texture'''

                