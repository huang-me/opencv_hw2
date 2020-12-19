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
	gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
	keypoints = detector.detect(gray)
	# show the result
	for i in range(len(keypoints)):
		image = cv.rectangle(	frame, 
								(round(keypoints[i].pt[0] - 5.5), round(keypoints[i].pt[1] - 5.5)),
								(round(keypoints[i].pt[0] + 5.5), round(keypoints[i].pt[1] + 5.5)), 
								(0, 0, 255),
								1)
		image = cv.line(	image, 
							(round(keypoints[i].pt[0]), round(keypoints[i].pt[1] - 5.5)),
							(round(keypoints[i].pt[0]), round(keypoints[i].pt[1] + 5.5)), 
							(0, 0, 255), 
							1)
		image = cv.line(	image, 
							(round(keypoints[i].pt[0] - 5.5), round(keypoints[i].pt[1])),
							(round(keypoints[i].pt[0] + 5.5), round(keypoints[i].pt[1])), 
							(0, 0, 255), 
							1)
	plt.ion()
	plt.imshow(image)
	plt.show()

def btn2_2():
	cap = cv.VideoCapture('Q2_Image/opticalFlow.mp4')

	lk_params = dict(	winSize=(15, 15),
						maxLevel=2,
						criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

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
	
	ret, last_frame = cap.read()
	detector = cv.SimpleBlobDetector_create(params)
	keypoints = detector.detect(last_frame)
	p0 = np.zeros((7, 1, 2), dtype='float32')

	for i in range(len(keypoints)):
		p = np.array([[round(keypoints[i].pt[0]), round(keypoints[i].pt[1])]], dtype='float32')
		p0[i] = p
	
	last_gray = cv.cvtColor(last_frame, cv.COLOR_BGR2GRAY)

	w = plt.imshow(last_gray)
	plt.ion()

	mask = np.zeros_like(last_frame)

	while ret:
		ret, frame = cap.read()
		if not ret:
			break
		gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
		
		p1, st, err = cv.calcOpticalFlowPyrLK(last_gray, gray, p0, None, **lk_params)
		good_new = p1[st == 1]
		good_old = p0[st == 1]
		for i, (new, old) in enumerate(zip(good_new, good_old)):
			a, b = new.ravel()
			c, d = old.ravel()
			mask = cv.line(mask, (a, b), (c, d), (0, 0, 255), 2)
			frame = cv.circle(frame, (a, b), 5, (0, 0, 255), -1)
		img = cv.add(frame, mask)
		w.set_data(img)
		plt.show()
		plt.pause(.001)
		# Now update the previous frame and previous points
		last_gray = gray.copy()
		p0 = good_new.reshape(-1, 1, 2)
	
	cap.release()