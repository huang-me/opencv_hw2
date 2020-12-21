from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

from mainwindow import Ui_MainWindow as ui
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import sys

from block1 import *
from block2 import *
from block3 import *
from block4 import *

class window(QMainWindow, ui):
	def __init__(self):
		super().__init__()
		self.setupUi(self)
		self.BTN_1.clicked.connect(btn1)
		self.BTN_2_1.clicked.connect(btn2_1)
		self.BTN_2_2.clicked.connect(btn2_2)
		self.BTN_3.clicked.connect(btn3)
		self.BTN_4_1.clicked.connect(btn4_1)
		self.BTN_4_2.clicked.connect(btn4_2)

if __name__ == '__main__':
	app = QApplication(sys.argv)
	win = window()
	win.show()
	sys.exit(app.exec_())
