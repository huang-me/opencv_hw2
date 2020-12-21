from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import os

def btn4_1():
    # find all images in folder
    imglist = os.listdir('Q4_Image')
    # setup subplot
    plt.ion()
    fig, axes = plt.subplots(4, 17, figsize=(17, 4))
    i = 0
    # show all images (before and after)
    for name in imglist:
        img = cv.imread(os.path.join('Q4_Image', name))
        # show original image
        row = int(i/17) if i < 17 else int(i/17) + 1
        axes[row][i%17].imshow(img)
        axes[row][i%17].axis('off')
        # PCA
        pca = PCA(n_components=100)
        shape = img.shape
        img_r = np.reshape(img, (shape[0], shape[1] * shape[2]))
        reduced = pca.fit_transform(img_r)
        res = pca.inverse_transform(reduced)
        res = np.reshape(res, shape)
        res = res.astype(np.uint8)
        row = int(i/17) + 1 if i < 17 else int(i/17) + 2
        axes[row][i%17].imshow(res)
        axes[row][i%17].axis('off')
        # next image
        i += 1
    plt.show()

def btn4_2():
    # find all images in folder
    imglist = os.listdir('Q4_Image')
    # initialize values
    error = np.zeros(34)
    i = 0
    # show all images (before and after)
    for name in imglist:
        img = cv.imread(os.path.join('Q4_Image', name))
        # PCA
        pca = PCA(n_components=100)
        shape = img.shape
        img_r = np.reshape(img, (shape[0], shape[1] * shape[2]))
        reduced = pca.fit_transform(img_r)
        res = pca.inverse_transform(reduced)
        res = np.reshape(res, shape)
        tmp =   abs(img[:,:,0] - res[:,:,0]) * 0.299 + \
                abs(img[:,:,1] - res[:,:,1]) * 0.587 + \
                abs(img[:,:,2] - res[:,:,2]) * 0.114
        error[i] = tmp.sum()
        i += 1
    print(error)