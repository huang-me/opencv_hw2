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
		from block2 import btn2_1
		btn2_1()

if __name__ == '__main__':
	app = QApplication(sys.argv)
	win = window()
	win.show()
	sys.exit(app.exec_())
