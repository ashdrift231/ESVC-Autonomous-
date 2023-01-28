import dlib as db 
from math import hypot
import numpy as np
import cv2
import os

class FaceMovDetector:
    def __init__(self):
        self.detector = db.get_frontal_face_detector() #loading the face detector
        self.predictor = db.shape_predictor("shape_predictor_68_face_landmarks.dat") #loading facial landmark model
        #setting flag variables
        self.blink = 0
        self.speak = 0
        os.environ["QT_QPA_PLATFORM"] = "xcb"

    def midpoint(self, p1, p2):
        return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

    def get_blinking_ratio(self, eye_points, facial_landmarks,frame):
        left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
        right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
        center_top = self.midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
        center_bottom = self.midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))
        hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
        ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)
        hor_line_len = hypot((left_point[0]- right_point[0]),(left_point[1]-right_point[1]))
        ver_line_len = hypot((center_top[0]-center_bottom[0]),(center_top[1]-center_bottom[1]))
        ratio = hor_line_len / ver_line_len
        return ratio

    def get_mouth_mov_ratio(self, mouth_points, facial_landmarks,frame):
        center_top_pt = (facial_landmarks.part(mouth_points[1]).x, facial_landmarks.part(mouth_points[1]).y)
        center_bottom_pt = (facial_landmarks.part(mouth_points[3]).x, facial_landmarks.part(mouth_points[3]).y)
        corner_left = (facial_landmarks.part(mouth_points[0]).x, facial_landmarks.part(mouth_points[0]).y)
        corner_right = (facial_landmarks.part(mouth_points[2]).x, facial_landmarks.part(mouth_points[2]).y)
        center_ver_line = cv2.line(frame, center_top_pt, center_bottom_pt, (0, 255, 0), 2)
        center_hor_line = cv2.line(frame, corner_left, corner_right, (0,255, 0),2)
        ver_line_len = hypot((center_top_pt[0]-center_bottom_pt[0]),(center_top_pt[1] - center_bottom_pt[1]))
        hor_line_len = hypot((corner_left[0]-corner_right[0]),(corner_left[1] - corner_right[1]))
        ratio = hor_line_len / ver_line_len
        return ratio
    
    def detect_mov(self):
        # specify the path of the video file and create a VideoCapture object
        cap = cv2.VideoCapture("/home/ashdrift/Desktop/detect liveness/output.mp4")
        # check if the video file is opened successfully
        if not cap.isOpened():
            print("Error opening video file")
        while True: 
            _, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray)
            for face in faces:
                landmarks = self.predictor(gray, face)
                left_eye_ratio = self.get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks,frame)
                right_eye_ratio = self.get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks,frame)
                mouth_ratio = self.get_mouth_mov_ratio([48,51,54,57], landmarks,frame)
                if left_eye_ratio > 5.7 or right_eye_ratio > 5.7:
                    self.blink = 1 #setting flag
                if mouth_ratio > 3.5:
                    self.speak = 1 #setting flag
            cv2.imshow("Video", frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            else:
                break   

    def ReturnFlag(self): #return flag
        return self.blink,self.speak






       
            