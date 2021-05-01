#import the necessary packages
import numpy as np
import cv2 as cv
import argparse as ap
import imutils

#Directly specify the path to the video
parser = ap.ArgumentParser()
parser.add_argument("-c", "--Camera")
parser.add_argument("-s", "--Size", type=int, default=64)
args = vars(parser.parse_args())

#define a range of given colors in HSV(hue-saturation-value)
#set range for  colors
lower_color = {'black': (10, 10, 10), 'green': (66, 122, 129), 'blue': (97, 100, 117), 'white': (200, 200, 200),
               'yellow': (23, 59, 119), 'red': (155, 25, 0)}
upper_color = {'black': (50, 55, 55), 'green': (86, 255, 255), 'blue': (117, 255, 255), 'white': (250, 250, 250),
               'yellow': (54, 255, 255), 'red': (179, 255, 255)}

colors = {'black': (0, 0, 0), 'green': (0, 255, 0), 'blue': (255, 0, 0), 'white': (255, 255, 255),
          'yellow': (0, 255, 217), 'red': (0, 0, 255)}

#take a link to a webcam
if not args.get("c", False):
#webcamera no 0 is used to capture the frames
    cam = cv.VideoCapture(0)

else:
    cam = cv.VideoCapture(args["c"])

#loop over frames from the video file stream
while True:

#grab the frame from the threaded video file stream
    (grabbed, frame) = cam.read() 
#if the frame was not grabbed, then we have reached the end
    if args.get("c") and not grabbed:
        break

#resize the frame
    frame = imutils.resize(frame, width=650)

    blur = cv.GaussianBlur(frame, (9, 9), 0)
#convert BGR to HSV
    bgr_to_hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)

# construct a mask for the colors
    for key, value in upper_color.items():
        kernel = np.ones((7, 7), np.uint8)
#masking the image to find our color
        mask = cv.inRange(bgr_to_hsv, lower_color[key], upper_color[key])
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
#finding contours in mask image
        con = cv.findContours(mask.copy(), cv.RETR_EXTERNAL,
                              cv.CHAIN_APPROX_SIMPLE)[-2]
        center = None

        if len(con) > 0:

            m = max(con, key=cv.contourArea)
            ((x, y), radius) = cv.minEnclosingCircle(m)
            M = cv.moments(m)
            cx = int(M["m10"]/M["m00"])
            cy = int(M["m01"]/M["m00"])

            rect = cv.minAreaRect(m)
            box = cv.boxPoints(rect)
            box = np.int0(box)
            cv.drawContours(frame,[box],0,(0,0,0),2)

            if radius > 0.5:
                cv.circle(frame,(cx,cy),int(radius),colors[key],5)
                cv.putText(frame, key + " " + "color", (cx-15,cy-15), cv.FONT_HERSHEY_SIMPLEX, 0.6,
                            colors[key], 2)

#show the frame to our screen
    cv.imshow("Frame", frame)

    wk = cv.waitKey(5)
    if wk == 27:
        break

#release the captured frame
cam.release()
#destroys all windows
cv.destroyAllWindows()
