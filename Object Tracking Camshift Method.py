import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

ret, frame = cap.read()

x, y, w, h = 200, 200, 100, 50
track_window = (x, y, w, h)

roi = frame[y:y + h, x:x + w]
hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
mask = cv.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255)))

roi_hist = cv.calcHist([hsv_roi], [0], mask, [180], [0, 180])

cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)

term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)

cv.imshow('roi', roi)

while 1:
    ret, frame = cap.read()
    if ret == True:
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        dst = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
        ret, track_window = cv.CamShift(dst, track_window, term_crit)

        # x, y, w, h = track_window
        # final_img = cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        pts = cv.boxPoints(ret)
        print(pts)
        pts = np.int0(pts)
        final_img = cv.polylines(frame, [pts], True, (0, 255, 0), 2)
        cv.imshow('dst', dst)
        cv.imshow('final image', final_img)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
    else:
        break
