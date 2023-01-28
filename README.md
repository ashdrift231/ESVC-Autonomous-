# Face-Liveness
The code is divided into 3 modules(python files), namely
● DetectTexture.py : Checks the texture and predicts spoofing faces of the
input frames
● DetectEyeMouth.py : Monitors eye blinking and mouth movement in the
face of the input frame
● main.py : Executes individual approaches

![packages](https://user-images.githubusercontent.com/47792802/215265504-23b4886d-bdff-4dbe-9c79-aa9ce971b53b.jpg)
![classes](https://user-images.githubusercontent.com/47792802/215265516-9fcdaf2d-4f49-4ead-be1e-212ca43bd7f2.jpg)

1]DetectTexture.py:The code "DetectTexture.py" is a Python class called "FaceTonDetector" that uses OpenCV and TensorFlow 
to detect the liveness of a face in a video. It loads a pre-trained Haar cascade classifier for face detection 
and a pre-trained texture-based liveness detection model. 
It reads a video file, captures frames, detects faces, crops and resizes them, and uses the pre-trained model to make a prediction. 
It sets a flag to 1 if the prediction is less than 0.5 and returns the flag value to check whether the face is live or spoofed.

2] DetectEyeMouth.py :This code is a Python class called "FaceMovDetector" 
that uses dlib library to detect liveness of a face in a video by detecting blink and mouth movement. 
The class has an init function that loads necessary models and libraries, including a pre-trained face detector 
and facial landmark predictor. 
It also initializes flags "blink" and "speak" with value 0. 
The class has functions to detect blink and mouth movement ratios, and a detect_mov() function 
that reads a video, captures frames, detects faces and landmarks, calculates blink and mouth movement ratios, 
and sets the flags. The frames are displayed on screen and the loop can be exited by pressing 'q'.

3]main.py : The code defines a class "Operator" that uses two other classes, "FaceMovDetector" and "FaceTonDetector",
to detect the liveness of a face in a video by analyzing blink and mouth movement, and texture-based features.
It creates instances of the other classes, initializes flag variables, and has functions to detect, 
record and analyze faces using multiprocessing and openCV library. At the end, it creates an instance of the class 
and calls its functions in a specific order.
