# import the necessary packages
from imutils import paths
import argparse
import cv2
import sys
import numpy as np
import os


def variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    return cv2.Laplacian(image, cv2.CV_64F).var()


img_path = '../text-detection-fots.pytorch/Data/low_quality_rejector/medium/'
# Anno path is used for verify detection results
anno_path = '../text-detection-fots.pytorch/Data/low_quality_rejector/res_medium/'
save_txt = "medium_roi.txt"
threshold = 150

# loop over the input images
file_object = open(save_txt, "w")
blur = 0
non_blur = 0
focus_measure = []
for imagePath in paths.list_images(img_path):
    # load the image, convert it to grayscale, and compute the
    # focus measure of the image using the Variance of Laplacian
    # method
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    base = os.path.split(imagePath)
    #     if os.path.exists(os.path.join(anno_path, 'res_' + base[1][:-4] + '.txt')):
    bboxes = np.loadtxt(os.path.join(anno_path, 'res_' + base[1][:-4] + '.txt'), dtype=int, delimiter=',')
    # IF bounding box available, apply laplacian on the roi region
    if bboxes.shape[0] >= 1:
        x1, y1, x2, y2 = np.amin(bboxes[::2]), np.amax(bboxes[1::2]), np.amax(bboxes[::2]), np.amin(bboxes[1::2])
        x1 = np.clip(x1, 0, gray.shape[1])
        y1 = np.clip(y1, 0, gray.shape[0])
        x2 = np.clip(x2, 0, gray.shape[1])
        y2 = np.clip(y2, 0, gray.shape[0])
        # Parse the bboxes
        roi = gray[y2:y1, x1:x2]
    else:
        roi = gray
    fm = variance_of_laplacian(roi)
    focus_measure.append(fm)

    if fm > threshold:
        text = imagePath + " - Not Blurry: " + str(fm)
        file_object.write(imagePath + " - Not Blurry: " + str(fm) + '\n')
        non_blur += 1
    #         print(imagePath+" - Not Blurry: "+str(fm))

    # if the focus measure is less than the supplied threshold,
    # then the image should be considered "blurry"
    if fm < threshold:
        text = imagePath + " - Blurry: " + str(fm)
        file_object.write(imagePath + " - Blurry: " + str(fm) + '\n')
        blur += 1
#         print(imagePath+" - Blurry: "+str(fm))

print("Blur images number: ", blur)
print("non blur images number:", non_blur)
file_object.close()