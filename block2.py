import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

def btn2_1():
	cap = cv.VideoCapture('Q2_Image/opticalFlow.mp4')
	# setup all parameters
	params = cv.SimpleBlobDetector_Params()
	params.minThreshold = 1
	params.maxThreshold = 255
	params.filterByArea = True
	params.minArea = 40
	params.maxArea = 60
	params.filterByCircularity = True
	params.minCircularity = 0.8
	params.filterByConvexity = True
	params.minConvexity = 0.8
	params.filterByInertia = True
	params.minInertiaRatio = 0.4
	# detect blobs in the image
	detector = cv.SimpleBlobDetector_create(params)
	ret, frame = cap.read()
	frame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
	keypoints = detector.detect(frame)
	new_img = cv.drawKeypoints(	frame, 
								keypoints, 
								np.array([]),
								(0, 0, 255), 
								cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	# show the result
	plt.ion()
	w = plt.imshow(new_img)
	plt.show()