import cv2
from DetectEyeMouth import FaceMovDetector
from DetectTexture import FaceTonDetector
import time
#import subprocess
import multiprocessing

class Operator():
    def __init__(self):
        self.ftd = FaceTonDetector() #initiate class objects
        self.fmd = FaceMovDetector()
        self.blink = 0
        self.speak = 0
        self.texture = 0

    def detect(self):
        if self.blink == 1 or self.speak == 1 or self.texture == 1:
            print("live face detected")
        else:
            print("spoof face detected")

    def get_recording(self):
        video = cv2.VideoCapture(0)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('output.mp4',fourcc, 20.0, (640,480))
        start_time = time.time()  # start the timer

        while (time.time() - start_time) < 10:
            ret, frame = video.read()
            if ret==True:
                out.write(frame)
                cv2.imshow('frame',frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        video.release()
        out.release() 
        cv2.destroyAllWindows()

    def analyze_faces(self):
        # Use Process to create a subprocess for the fmd.detect_mov() method
        a1 = multiprocessing.Process(target=self.fmd.detect_mov)
        a1.start()
        a1.join()
        # Use Process to create a subprocess for the ftd.detect_tone() method
        a2 = multiprocessing.Process(target=self.ftd.detect_tone)
        a2.start()
        a2.join()
        self.blink , self.speak = self.fmd.ReturnFlag()
        self.texture = self.ftd.FlagReturn()

if __name__ == "__main__":
    op = Operator()
    op.get_recording()
    op.analyze_faces()
    op.detect() 


    '''a1 = multiprocessing.Process(target=fmd.detect_mov())
a1.start()

a2 = multiprocessing.Process(target=ftd.detect_tone())
a2.start()'''


'''# Use Popen to create a subprocess for the fmd.detect_mov() method
process1 = subprocess.Popen(['python', '-c', 'fmd.detect_mov()'], stdout=subprocess.PIPE)

# Use Popen to create a subprocess for the ftd.detect_tone() method
process2 = subprocess.Popen(['python', '-c', 'ftd.detect_tone()'], stdout=subprocess.PIPE)

# Wait for the first subprocess to finish
process1.wait()

# Wait for the second subprocess to finish
process2.wait()'''

'''def analyze_faces(self):
        ftd = FaceTonDetector()
        fmd = FaceMovDetector()
        # Use Popen to create a subprocess for the fmd.detect_mov() method
        a1 = subprocess.Popen(['python', '-c', '{}.detect_mov()'.format(fmd)], stdout=subprocess.PIPE)

        # Use Popen to create a subprocess for the ftd.detect_tone() method
        a2 = subprocess.Popen(['python', '-c', '{}.detect_tone()'.format(ftd)], stdout=subprocess.PIPE)

        # Wait for the first subprocess to finish
        a1.wait()

        # Wait for the second subprocess to finish
        a2.wait()
        self.blink , self.speak = fmd.ReturnFlag()
        self.texture = ftd.FlagReturn()'''