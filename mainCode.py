from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import *
from threading import Timer
import RPi.GPIO as GPIO
from ctypes import *
import numpy as np
import argparse
import threading
import imutils
import ctypes
import time
import dlib
import cv2

lib = ctypes.cdll.LoadLibrary("./libhaisantts.so")
lib.startHaisanTTS.argtypes=[POINTER(c_char)]

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

LED = 4     #physical 7
PIN = 17    #physical 11

GPIO.setup(LED,GPIO.OUT)
GPIO.setup(PIN,GPIO.IN,pull_up_down = GPIO.PUD_DOWN)
GPIO.add_event_detect(PIN,GPIO.RISING)
    
def ClockWarning(msg):
    TTS=(c_char * 100)(*bytes(msg,'utf-8'))
    cast(TTS, POINTER(c_char))
    lib.startHaisanTTS(TTS)

def LTWarning():
    ClockWarning("开了很久了，去休息吧！")

class LoopTimer(Timer):  

    def __init__(self, interval, function, args=[], kwargs={}):
        Timer.__init__(self,interval, function, args, kwargs)  
 
def WarningLed():
    print("led binked")

def warning(msg):
    global alarm_status
    global alarm_status2
    global saying
    while alarm_status:
        print('call')
        TTS=(c_char * 100)(*bytes(msg,'utf-8'))
        cast(TTS, POINTER(c_char))
        lib.startHaisanTTS(TTS)
    if alarm_status2:
        print('call')
        saying = True
        TTS=(c_char * 100)(*bytes(msg,'utf-8'))
        cast(TTS, POINTER(c_char))
        lib.startHaisanTTS(TTS)
        saying = False

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def final_ear(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)
    ear = (leftEAR + rightEAR) / 2.0
    return (ear, leftEye, rightEye)

def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))
    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))
    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)
    distance = abs(top_mean[1] - low_mean[1])
    return distance

ap = argparse.ArgumentParser()
ap.add_argument("--webcam", type=int, default=0,help="index of webcam on system")
args = vars(ap.parse_args())
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 10
YAWN_THRESH = 25
alarm_status = False
alarm_status2 = False
saying = False
COUNTER = 0
print("-> Loading the predictor and detector...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
print("-> Starting Video Stream")
vs= VideoStream(usePiCamera=True).start()    
time.sleep(1.0)
t = LoopTimer(20.0, LTWarning)
t.start()
flag = True
start = 0
end = 0
time_dif = 0

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    for rect in rects: 
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        eye = final_ear(shape)
        ear = eye[0]
        leftEye = eye [1]
        rightEye = eye[2]

        distance = lip_distance(shape)

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        lip = shape[48:60]
        cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)

        if ear < EYE_AR_THRESH:
            COUNTER += 4
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                if alarm_status == False:
                    alarm_status = True
                    t1 = Thread(target=warning, args=('醒醒，醒醒！',))
                    t1.daemon = True
                    t1.start()

                cv2.putText(frame, "warning!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            COUNTER = 0
            alarm_status = False

        if (distance > YAWN_THRESH):
                cv2.putText(frame, "Drawning alart!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                if alarm_status2 == False and saying == False:
                    if flag == True:
                        start = time.time()
                        flag = False
                    elif flag == False:
                        end = time.time()
                        time_dif = end - start
                        flag = True
                        if time_dif < 20:
                            GPIO.output(LED,GPIO.HIGH)
                    print(flag)
                    alarm_status2 = True
                    t2 = Thread(target=warning, args=('困了吧，停车休息一下吧！',))
                    t2.daemon = True
                    t2.start()
        else:
            alarm_status2 = False

        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "YAWN: {:.2f}".format(distance), (300, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        time.sleep(1)
        break
    if GPIO.event_detected(PIN):
        GPIO.output(LED,GPIO.LOW)
        continue
    
cv2.destroyAllWindows()
GPIO.output(LED,GPIO.LOW)
vs.stop()

