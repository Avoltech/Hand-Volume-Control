import cv2
import mediapipe as mp
import time
import numpy as np

class HandDetector():
    def __init__(self, static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            self.static_image_mode,
            self.max_num_hands,
            self.min_detection_confidence,
            self.min_tracking_confidence
        )

        self.mp_draw = mp.solutions.drawing_utils
        
    def findHands(self, img, draw=True):
        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_RGB)

        if self.results.multi_hand_landmarks:
            for hand_lms in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, hand_lms, self.mp_hands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, hand_number=0, draw=True):
        lm_list = []
        bound_coords = [0, 0, 0, 0]
        if self.results.multi_hand_landmarks:
            selected_hand = self.results.multi_hand_landmarks[hand_number]
            for id, lm in enumerate(selected_hand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                #print(id, cx, cy)
                lm_list.append([id, cx, cy])

                if draw:
                    cv2.circle(img, (cx, cy), 3, (0, 255, 255), -1)
            arr = np.array(lm_list)
            xmin = arr[:, 1].min()
            ymin = arr[:, 2].min()
            xmax = arr[:, 1].max()
            ymax = arr[:, 2].max()
            bound_coords = [xmin, ymin, xmax, ymax]
        return lm_list, bound_coords



def main():
    cap = cv2.VideoCapture(0)

    cur_time = 0
    prev_time = 0

    detector = HandDetector()
    while True:
        _,frame = cap.read()

        img = detector.findHands(frame)
        img = detector.findPosition(frame)
        c_time = time.time()
        fps = 1/ (c_time - prev_time)
        prev_time = c_time

        cv2.putText(frame, "fps: "+str(int(fps)), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("frame", frame)
      
        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    main()