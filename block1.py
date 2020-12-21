import cv2 as cv
import numpy as np

def btn1():
    cap = cv.VideoCapture('Q1_Image/bgSub.mp4')
    zeros = np.zeros((176, 320, 50))
    means = np.zeros((176, 320))
    std = np.zeros((176, 320))

    if cap.isOpened():
        for i in range(50):
            ret, frame = cap.read()
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            zeros[:, :, i] = frame_gray
        for i in range(176):
            for j in range(320):
                means[i, j] = zeros[i, j, :].mean()
                std[i, j] = zeros[i, j, :].std()
                if std[i, j] < 5:
                    std[i, j] = 5

        while ret:
            ret, frame = cap.read()
            if ret == 0:
                break
            new_frame = np.zeros((176, 320, 3), dtype='uint8')
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            for i in range(176):
                for j in range(320):
                    if abs(frame_gray[i, j] - means[i, j]) > std[i, j] * 5:
                        new_frame[i, j, :] = 255


            hstack = np.hstack((frame, new_frame))
            cv.imshow('background substraction', hstack)
            if cv.waitKey(1) == ord('q'):
                break
    cap.release()
    cv.destroyAllWindows()