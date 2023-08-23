import numpy as np
import cv2
from deepface import DeepFace

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError('Cannot open Webcam')


# face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

font = cv2.FONT_HERSHEY_SIMPLEX

while True:
	ret, frame = cap.read()
    
	result = DeepFace.analyze(frame, actions = ['emotion'])

	# Convert frame to grayscale
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Detect multi faces in the image
	faces = face_cascade.detectMultiScale(gray, 1.1, 4)

	for face in faces:
		x, y, w, h = face
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

		# Draw rectangle in the original image
		cv2.putText(frame, result['dominant_emotion'],(50,50), cv2.FONT_HERSHEY_SIMPLEX, 3,(255,255,0),2,cv2.LINE_AA)
		# cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,100), 2)

	cv2.imshow("Faces", frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()