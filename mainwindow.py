# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'hw2.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(450, 530)
        self.group1 = QtWidgets.QGroupBox(MainWindow)
        self.group1.setGeometry(QtCore.QRect(20, 30, 400, 70))
        self.group1.setObjectName("group1")
        self.BTN_1 = QtWidgets.QPushButton(self.group1)
        self.BTN_1.setGeometry(QtCore.QRect(30, 30, 340, 30))
        self.BTN_1.setObjectName("BTN_1")
        self.group2 = QtWidgets.QGroupBox(MainWindow)
        self.group2.setGeometry(QtCore.QRect(20, 110, 400, 110))
        self.group2.setObjectName("group2")
        self.BTN_2_1 = QtWidgets.QPushButton(self.group2)
        self.BTN_2_1.setGeometry(QtCore.QRect(30, 30, 340, 30))
        self.BTN_2_1.setObjectName("BTN_2_1")
        self.BTN_2_2 = QtWidgets.QPushButton(self.group2)
        self.BTN_2_2.setGeometry(QtCore.QRect(30, 70, 340, 30))
        self.BTN_2_2.setObjectName("BTN_2_2")
        self.group3 = QtWidgets.QGroupBox(MainWindow)
        self.group3.setGeometry(QtCore.QRect(20, 230, 400, 70))
        self.group3.setObjectName("group3")
        self.BTN_3 = QtWidgets.QPushButton(self.group3)
        self.BTN_3.setGeometry(QtCore.QRect(30, 30, 340, 30))
        self.BTN_3.setObjectName("BTN_3")
        self.group4 = QtWidgets.QGroupBox(MainWindow)
        self.group4.setGeometry(QtCore.QRect(20, 310, 400, 110))
        self.group4.setObjectName("group4")
        self.BTN_4_1 = QtWidgets.QPushButton(self.group4)
        self.BTN_4_1.setGeometry(QtCore.QRect(30, 30, 340, 30))
        self.BTN_4_1.setObjectName("BTN_4_1")
        self.BTN_4_2 = QtWidgets.QPushButton(self.group4)
        self.BTN_4_2.setGeometry(QtCore.QRect(30, 70, 340, 30))
        self.BTN_4_2.setObjectName("BTN_4_2")
        self.group5 = QtWidgets.QGroupBox(MainWindow)
        self.group5.setGeometry(QtCore.QRect(20, 430, 400, 70))
        self.group5.setObjectName("group5")
        self.BTN_5 = QtWidgets.QPushButton(self.group5)
        self.BTN_5.setGeometry(QtCore.QRect(30, 30, 340, 30))
        self.BTN_5.setObjectName("BTN_5")

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Dialog"))
        self.group1.setTitle(_translate("MainWindow", "1. Background Subtraction"))
        self.BTN_1.setText(_translate("MainWindow", "1.1 Background Subtraction"))
        self.group2.setTitle(_translate("MainWindow", "2. Optical Flow"))
        self.BTN_2_1.setText(_translate("MainWindow", "2.1 Preprocessing"))
        self.BTN_2_2.setText(_translate("MainWindow", "2.2 Video Tracking"))
        self.group3.setTitle(_translate("MainWindow", "3. Perspective Transform"))
        self.BTN_3.setText(_translate("MainWindow", "3.1 Perspective Transform"))
        self.group4.setTitle(_translate("MainWindow", "4. PCA"))
        self.BTN_4_1.setText(_translate("MainWindow", "4.1 Image Reconstruction"))
        self.BTN_4_2.setText(_translate("MainWindow", "4.2 Compute the Reconstruction Error"))
        self.group5.setTitle(_translate("MainWindow", "5. Dogs and Cats classifacation using ResNet50"))
        self.BTN_5.setText(_translate("MainWindow", "5.1 Dogs and Cats classification"))
