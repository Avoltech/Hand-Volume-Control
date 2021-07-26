import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

#used for drawing hand points
mp_draw = mp.solutions.drawing_utils

#current time, previous time, for calciulating fps
cur_time = 0
prev_time = 0

while True:
	_, frame = cap.read()
	#convert image to RGB, because meadipipe uses RGB input
	img_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	#detect hands
	results = hands.process(img_RGB)

	if results.multi_hand_landmarks:
		for hand_lms in results.multi_hand_landmarks:
			mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

	c_time = time.time()
	fps = 1/ (c_time - prev_time)
	prev_time = c_time

	cv2.putText(frame, "fps: "+str(int(fps)), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
	cv2.imshow("frame", frame)

	if cv2.waitKey(1) == ord('q'):
		cv2.destroyAllWindows()
		break
