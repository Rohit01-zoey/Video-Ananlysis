import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)


fgbg = cv.createBackgroundSubtractorMOG2(detectShadows=False)

while 1:
    ret, frame = cap.read()
    if frame is None:
        break

    fgmask = fgbg.apply(frame)
    cv.imshow('frame', frame)
    cv.imshow('fg mask frame', fgmask)

    k = cv.waitKey(30)
    if k == 27:
        break

cv.destroyAllWindows()
cap.release()
