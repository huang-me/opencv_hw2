from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

from mainwindow import Ui_MainWindow as ui
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import sys

class window(QMainWindow, ui):
	def __init__(self):
		super().__init__()
		self.setupUi(self)
		self.BTN_2_1.clicked.connect(self.btn2_1)
	
	def btn2_1(self):
		cap = cv.VideoCapture('Q2_Image/opticalFlow.mp4')
		params = cv.SimpleBlobDetector_Params()
		params.filterByCircularity = True
		params.minCircularity = 0.7
		params.filterByConvexity = True
		params.minConvexity = 0.5
		detector = cv.SimpleBlobDetector_create(params)
		ret, frame = cap.read()
		plt.ion()
		w = plt.imshow(frame)
		plt.show()
		while(ret):
			# frame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
			keypoints = detector.detect(frame)
			new_img = cv.drawKeypoints(	frame, 
										keypoints, 
										np.array([]),
										(0, 0, 255), 
										cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
			w.set_data(new_img)
			plt.show()
			plt.pause(0.01)
			ret, frame = cap.read()

if __name__ == '__main__':
	app = QApplication(sys.argv)
	win = window()
	win.show()
	sys.exit(app.exec_())
