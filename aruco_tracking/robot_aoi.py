from imutils.video import VideoStream
import numpy as np
import cv2 

from flask import Response
from flask import Flask
#from flask import render_template
import time
import imutils
import threading
#import argparse
#import datetime

outputFrame = None
lock = threading.Lock()

#app = Flask(__name__)

arucoDict   = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
arucoParams = cv2.aruco.DetectorParameters()
detector    = cv2.aruco.ArucoDetector(arucoDict, arucoParams)

vs = VideoStream(src=2).start()
time.sleep(2.0)

def processVideo():
  global vs, outputFrame, lock, detector
  # corners of the bigger rectangle
  topLeftBig = (0,0)
  topRightBig = (0,0)
  bottomLeftBig = (0,0)
  bottomRightBig = (0,0)

  GRID_SIZE = 80

  while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 1000 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=1000)

    # detect ArUco markers in the input frame
    corners, ids, rejected = detector.detectMarkers(frame)  

    # verify *at least* one ArUco marker was detected
    if len(corners) > 0:
    #if len(corners) == 4:
      # flatten the ArUco IDs list
      ids = ids.flatten()

      # loop over the detected ArUCo corners
      for (markerCorner, markerID) in zip(corners, ids):
        # extract the marker corners (which are always returned
        # in top-left, top-right, bottom-right, and bottom-left
        # order)
        (topLeft, topRight, bottomRight, bottomLeft) = markerCorner.reshape((4, 2))

        # convert each of the (x, y)-coordinate pairs to integers
        topLeft     = (int(topLeft[0]), int(topLeft[1]))
        #topRight    = (int(topRight[0]), int(topRight[1]))
        bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
        #bottomLeft  = (int(bottomLeft[0]), int(bottomLeft[1]))

        # draw the bounding box of the ArUCo detection
        #cv2.line(frame, topLeft, topRight, (0, 255, 0), 1)
        #cv2.line(frame, topRight, bottomRight, (0, 255, 0), 1)
        #cv2.line(frame, bottomRight, bottomLeft, (0, 255, 0), 1)
        #cv2.line(frame, bottomLeft, topLeft, (0, 255, 0), 1)

        # compute and draw the center (x, y)-coordinates of the
        # ArUco marker
        mcX = int((topLeft[0] + bottomRight[0]) / 2.0)
        mcY = int((topLeft[1] + bottomRight[1]) / 2.0)
        cv2.circle(frame, (mcX, mcY), 4, (0, 0, 255), -1)

        # collect coordinate of center for bigger rectangle
        match markerID:
          case 24: topLeftBig = [mcX,mcY]
          case 42: topRightBig = [mcX,mcY]
          case 66: bottomLeftBig = [mcX,mcY]
          case 70: bottomRightBig = [mcX,mcY]

        # draw the ArUco marker ID on the frame
        #cv2.putText(frame, str(markerID),
        #	(topLeft[0], topLeft[1] - 15),
        #	cv2.FONT_HERSHEY_SIMPLEX,
        #	0.5, (0, 255, 0), 2)

      # draw the bigger rectangle
      pts = np.array([topLeftBig, topRightBig, bottomRightBig, bottomLeftBig], np.int32)
      pts = pts.reshape((-1, 1, 2))
      cv2.polylines(frame, [pts], True, (0, 255, 0), 1)
        
      # show the output frame
      cv2.imshow("Video Source", frame)
	
      # prepare the source and destination points for geometric transformation
      sourcePts = np.float32([topLeftBig, topRightBig, bottomLeftBig, bottomRightBig])
      #sourcePts = np.float32(mCorners)
      destPts = np.float32([[0, 0], [800, 0], [0, 800], [800, 800]])

      # compute the perspective matrix
      matrix = cv2.getPerspectiveTransform(sourcePts, destPts)
      result = cv2.warpPerspective(frame, matrix, (800, 800))
			
      # Draw the navigation grid		
      tcX = tcY = int(GRID_SIZE / 2)
      height, width, _ = result.shape
		
      # draw tiles
      # draw vertical lines
      for x in range(0, width - 1, GRID_SIZE):
        cv2.line(result, (x, 0), (x, height), (0,255,0), 1, 1)

      # draw horizontal lines
      for y in range(0, height - 1, GRID_SIZE):
        cv2.line(result, (0, y), (width, y), (0,255,0), 1, 1)
		
      # draw red dots in the center of tiles
      for y in range(tcX, height - 1, GRID_SIZE):
        for x in range(tcX, width - 1, GRID_SIZE):
          cv2.circle(result, (x, y), 4, (0, 0, 255), -1)

      cv2.imshow("AOI Top View", result)
		
    else:
      cv2.imshow("Video Source", frame)

    # check if a key is pressed		
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
      break


#drawGridMap = False
#point = (0,0)

#tileSize = 80
#h = 800 # height
#w = 800 # width
#rect = []

#for y in range(0, h - 1, tileSize):
#  for x in range(0, w - 1, tileSize):
#    tL = (x,y)
#    bR = (x + (tileSize-1), y + (tileSize-1))
#    rect.append([tL,bR])

#tile_status = []
#for i in range(0, len(rect)):
#  tile_status.append(0)

#tileColor = [[(127,127,127),1],[(0,0,0),-1],[(0,255,255),-1], [(0,255,0),-1]]

#def is_inside_rectangle(point, rect):
#  (x, y) = point
#  (xmin, ymin), (xmax, ymax) = rect
#  return xmin <= x <= xmax and ymin <= y <= ymax

#def insideRect(rects, point):
#  size = len(rects)
#  for i in range(0, size):
#    if is_inside_rectangle(point, rects[i]): break
#  return i

#def gridview():	  
#  global vs, outputFrame, drawGridMap, point, lock
#  while True:
#    frame = vs.read()
#    result = imutils.resize(frame, width=800)

#    height, width, _ = result.shape

#    for x in range(0, w - 1, tileSize):
#      result = cv2.line(result, (x,0), (x,h), (255,255,255), 1)
#    for y in range(0, h - 1, tileSize):
#      result = cv2.line(result, (0,y), (w,y), (255,255,255), 1)

#    if drawGridMap:
#      i = insideRect(rect, point)
#      tile_status[i] = (tile_status[i]+1) % 4 # cycle between 0,1,2,3
#      drawGridMap   = False

    # draw tiles
#    for i in range(0, len(rect)):
#      if tile_status[i] > 0:
#        [color, fill] = tileColor[tile_status[i]]
#        cv2.rectangle(result,rect[i][0], rect[i][1], color, fill)

#    with lock:
#      outputFrame = result.copy()

#def generate():
  # grab global references to the output frame and lock variables
#  global outputFrame, lock
  # loop over frames from the output stream
#  while True:
    # wait until the lock is acquired
#    with lock:
      # check if the output frame is available, otherwise skip
      # the iteration of the loop
#      if outputFrame is None:
#        continue
      # encode the frame in JPEG format
#      (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
      # ensure the frame was successfully encoded
#      if not flag:
#        continue
    # yield the output frame in the byte format
#    yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
#      bytearray(encodedImage) + b'\r\n')

#@app.route("/video_feed")
#def video_feed():
  # return the response generated along with the specific media
  # type (mime type)
#  return Response(generate(),
#    mimetype = "multipart/x-mixed-replace; boundary=frame")

if __name__ == '__main__':
  processVideo()
#  thread = threading.Thread(target=gridview)
#  thread.daemon = True
#  thread.start()
#  app.run(host="0.0.0.0", port="8000", debug=True, threaded=True, use_reloader=False)

vs.stop()
cv2.destroyAllWindows()


