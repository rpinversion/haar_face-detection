# haar_face-detection
face detection with photo as well as live video
import cv2 as cv
import numpy as np
# two classifiers : haar cascades and local binary patterns      adelaides' face recognizer
# rudra = cv.imread('pictures/rudra.jpg')
# cv.imshow('rudra',rudra)
#blank = np.zeros(rudra.shape[:],dtype='uint8')

#gray = cv.cvtColor(rudra,cv.COLOR_BGR2GRAY)
#cv.imshow('gray',gray)
haar_cascade = cv.CascadeClassifier('haar_face.xml')
capture = cv.VideoCapture(0)
while True:
    isTrue, frame = capture.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    face_rect = haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=7)
#print(f'no of faces found = {len(face_rect)}')
    for (x,y,w,h) in face_rect:
        cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),thickness=2)
        cv.imshow('dete', frame)
    if cv.waitKey(20) & 0xFF == ord('e'):
        break
capture.release()
cv.destroyAllWindows()

    #cv.imshow('dete',rudra)
#print(face_rect)
# rectangle = cv.rectangle(blank.copy(),(353,157),(140,140),(0,255,0),thickness=1)
# cv.imshow('rectangle',rectangle)

cv.waitKey(0)
