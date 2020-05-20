#!/usr/bin/env python3

import cv2 as cv
import numpy as np
from pynput.mouse import Button,Controller
import wx
import imutils
#from imutils.video import WebcamVideoStream
from imutils.video import FPS

mouse = Controller()
app = wx.App(False)
(width, height) = wx.GetDisplaySize()
(camx,camy) = (320,240)

#camera = WebcamVideoStream(src=0).start()
camera = cv.VideoCapture(0)
camera.set(3, camx)
camera.set(4, camy)

fps = FPS().start()

mlocold = np.array([0, 0])
mouseloc = np.array([0, 0])

damfac = 4.5
pinch = False

old_position = (0,0)

while True:
    retval, image = camera.read()
    #image = camera.read()

    # Face Detection
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
    color = (0, 0, 0)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    area = 0
    X = Y = W = H = 0
    for (x, y, w, h) in faces:
        if w * h > area:
            area = w * h
            X, Y, W, H = x, y, w, h
    cv.rectangle(image, (X, Y), (X + W, Y + H), color)


    # Finger Detection 
    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv_image, np.array([33,80,40]), np.array([102,255,255]))
    mask_open = cv.morphologyEx(mask, cv.MORPH_OPEN, np.ones((5,5)))
    mask_close = cv.morphologyEx(mask_open, cv.MORPH_CLOSE, np.ones((20,20)))
    mask_final = mask_close
   
    contours,_ = cv.findContours(mask_final, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(image, contours, -1, (0,0,255), 2)
    if len(contours) == 2:
        if(pinch == True):
            pinch = False
            mouse.release(Button.left)

        x1,y1,w1,h1 = cv.boundingRect(contours[0]) 
        x2,y2,w2,h2 = cv.boundingRect(contours[1])
        cx1, cy1 = round((x1+w1/2)), round(y1+h1/2)
        cx2, cy2 = round((x2+w2/2)), round(y2+h2/2)

        cv.circle(image, (cx1,cy1), 2, (0,0,255), 2)
        cv.circle(image, (cx2,cy2), 2, (0,0,255), 2)
        cv.line(image, (cx1,cy1), (cx2,cy2), (255,0,0), 2)
        cx = round(cx1/2 + cx2/2)
        cy = round(cy1/2 + cy2/2)
        new_position = ((camx - cx)*damfac, cy*damfac)
        difference = np.sum([abs(new - old) for new, old in (new_position, old_position)])
        if difference > 10:
            mouse.position = new_position
            old_position = new_position
        #cv.rectangle(image,(x1,y1),(x1+w1,y1+h1),(255,0,0),2)
    elif len(contours) == 1:
        if pinch == False:    
            pinch = True
            mouse.click(Button.left, 1)

    flipHorizontal = cv.flip(image, 1)
    cv.imshow("Control thy screen", flipHorizontal)
    k = cv.waitKey(5)
    if k == ord('q'):
        break

    fps.update()

fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

