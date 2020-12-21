import cv2 as cv
import numpy as np

def btn3():
	cap = cv.VideoCapture('Q3_Image/test4perspective.mp4')
	im_src = cv.imread('Q3_Image/rl.jpg')
	fourcc = cv.VideoWriter_fourcc(*'mp4v')
	out = cv.VideoWriter('Perspective_transform.mp4', fourcc, 20.0, (1280, 720))

	white = np.zeros((2068, 3575, 3), np.uint8)
	white[:, :, 0] = np.zeros((2068, 3575)) + 255
	white[:, :, 1] = np.zeros((2068, 3575)) + 255
	white[:, :, 2] = np.zeros((2068, 3575)) + 255

	# get frame of the video
	hasFrame, frame = cap.read()

	while 1:
		# load dictionary used to generate the markers
		dictionary = cv.aruco.Dictionary_get(cv.aruco.DICT_6X6_250)
		# initialize the detector parameters using default values
		parameters = cv.aruco.DetectorParameters_create()
		# detect the markers in the image
		markerCorners, markerIds, rejectedCandidates = cv.aruco.detectMarkers(
			frame,
			dictionary,
			parameters = parameters
		)
		try:
			# find pixel value of the marker which's ID is 25 & 33
			index = np.array(np.squeeze(np.where(markerIds == 25)))
			refPt1 = np.squeeze(markerCorners[index[0]])[1]
			index = np.array(np.squeeze(np.where(markerIds == 33)))
			refPt2 = np.squeeze(markerCorners[index[0]])[2]

			distance = np.linalg.norm(refPt1 - refPt2)

			scalingFac = 0.02
			# calculate edge points for perspective
			pts_dst = [[refPt1[0] - round(scalingFac*distance), refPt1[1] - round(scalingFac*distance)]]
			pts_dst += [[refPt2[0] - round(scalingFac*distance), refPt2[1] - round(scalingFac*distance)]]
			index = np.array(np.squeeze(np.where(markerIds == 30)))
			refPt3 = np.squeeze(markerCorners[index[0]])[0]
			pts_dst += [[refPt3[0] - round(scalingFac*distance), refPt3[1] - round(scalingFac*distance)]]
			index = np.array(np.squeeze(np.where(markerIds == 23)))
			refPt4 = np.squeeze(markerCorners[index[0]])[0]
			pts_dst += [[refPt4[0] - round(scalingFac*distance), refPt4[1] - round(scalingFac*distance)]]
			pts_src = [ [0, 0], 
						[im_src.shape[1], 0],
						[im_src.shape[1], im_src.shape[0]], 
						[0, im_src.shape[0]]]
			pts_dst = np.array(pts_dst)
			pts_src = np.array(pts_src)

			retval, mask = cv.findHomography(pts_src, pts_dst)
			dst = cv.warpPerspective(im_src, retval, (1280, 720))
			mask = cv.warpPerspective(white, retval, (1280, 720))
			mask = cv.bitwise_not(mask)
			frame = cv.bitwise_and(frame, mask)
			dst = cv.bitwise_or(frame, dst)

			cv.imshow('Perspective Transform', dst)
			out.write(dst)
			if cv.waitKey(1) == ord('q'):
				break

		except:
			pass

		hasFrame, frame = cap.read()
		if not hasFrame:
			break

	cap.release()
	out.release()
	cv.destroyAllWindows()