import cv2
import HandTrackModule as htm
import time
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import numpy as np

detector = htm.HandDetector(min_detection_confidence=0.8, min_tracking_confidence=0.8)

cap = cv2.VideoCapture(0)
prev_time = 0
curr_time = 0

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
#volume.GetMute()
#volume.GetMasterVolumeLevel()
min_vol = volume.GetVolumeRange()[0]
max_vol = volume.GetVolumeRange()[1]
#volume.SetMasterVolumeLevel(-20.0, None)

LITTLE_FINGER_DOWN = False
LITTLE_FINGER_UP = True
AUDIO_CONTROL_FLAG = False

while True:
    _, frame = cap.read()
    img = detector.findHands(frame)


    lm_list, bound_coords = detector.findPosition(img, draw=False)
    
    
    if lm_list:

        little_finder_top_link_length = lm_list[19][2] - lm_list[20][2]
        middle_finder_top_link_length = lm_list[11][2] - lm_list[12][2]

        if little_finder_top_link_length < 0.2*middle_finder_top_link_length:
            AUDIO_CONTROL_FLAG = True
        else:
            AUDIO_CONTROL_FLAG = False

        #print(little_finder_top_link_length)

        if AUDIO_CONTROL_FLAG:
        

            #for scaling finger distance
            cal_x1, cal_y1 = lm_list[17][1], lm_list[17][2]
            cal_x2, cal_y2 = lm_list[0][1], lm_list[0][2]
            cal_length = math.hypot((cal_x2-cal_x1), (cal_y2-cal_y1))

            
            x1, y1 = lm_list[8][1], lm_list[8][2]
            x2, y2 = lm_list[4][1], lm_list[4][2]   
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            cv2.line(img, (x1, y1), (x2, y2), (225, 0, 255), 5)
            cv2.circle(img, (x1, y1), 7, (255, 0, 0), -1)
            cv2.circle(img, (x2, y2), 7, (225, 0, 0), -1)
            cv2.circle(img, (cx, cy), 7, (225, 0, 0), -1)

            length = math.hypot((x2-x1), (y2-y1))


            if length < 0.15:
                cv2.circle(img, (cx, cy), 7, (225, 255, 0), -1)

            height_wrt_cal_length = img.shape[0] / cal_length
            #print(height_wrt_cal_length)

            vol = np.interp(height_wrt_cal_length * length, [125, 850], [min_vol, max_vol])
            volPer = np.interp(height_wrt_cal_length * length, [125, 850], [0, 100])
            smoothness = 5
            smoothVol = smoothness * round(volPer/smoothness)
            volSet = ((max_vol - min_vol) * smoothVol/100) + min_vol
            print(vol)
            volume.SetMasterVolumeLevel(volSet, None)
   
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(img, "FPS: "+str(int(fps)), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("frame", img)
    
    if cv2.waitKey(1) == ord('q'):
        cv2.destroyAllWindows()
        break


