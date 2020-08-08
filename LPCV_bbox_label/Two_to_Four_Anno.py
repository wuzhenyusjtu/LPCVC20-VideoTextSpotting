import os
import cv2
from PyQt5.QtGui import QImage
from read_write_xml import PascalVocWriter, parse_rec

# The purpose for this file is to change two points annotation to four points
cnt = 0
base_dir = "/home/johnhu/Pictures/images/"
output_dir = "/home/johnhu/Pictures/modified_anno/"
annotation_dir = "/home/johnhu/Pictures/original_anno/"

for filename in os.listdir(base_dir):
    # counter
    cnt = cnt + 1
    print(cnt)

    image = cv2.imread(os.path.join(base_dir, filename))
    annotation_filename = filename.replace(".jpg", ".xml")

    if os.path.exists(os.path.join(annotation_dir, annotation_filename)):
        annotation_path = os.path.join(annotation_dir, annotation_filename)
        bboxes = parse_rec(annotation_path)
        imagePath = os.path.join(base_dir, filename)
        imgFolderPath = os.path.dirname(imagePath)
        imgFolderName = os.path.split(imgFolderPath)[-1]
        imgFileName = os.path.basename(imagePath)
        bbox_image = QImage()
        bbox_image.load(os.path.join(base_dir, filename))
        imageShape = [bbox_image.height(), bbox_image.width(), 1 if bbox_image.isGrayscale() else 3]
        writer = PascalVocWriter(imgFolderName, imgFileName, imageShape, localImgPath=imagePath)
        writer.verified = False

        # Parse the original bbox, and initiaize
        bboxes = parse_rec(annotation_path)

        # Original annotation bbox
        bb1 = []
        for bbox in bboxes:
            bbox_coordinates = bbox['bbox']
            x_min = bbox_coordinates[0]
            y_min = bbox_coordinates[1]
            x_max = bbox_coordinates[2]
            y_max = bbox_coordinates[3]
            bb1.append([x_min, y_min, x_min, y_max, x_max, y_max, x_max, y_min])

        for i in range(0, len(bb1)):
            label = bboxes[i]['name']
            difficult = bboxes[i]['difficult']
            x1, y1, x2, y2, x3, y3, x4, y4 = bb1[i]
            writer.addBndBox(x1, y1, x2, y2, x3, y3, x4, y4, label, difficult)

        writer.save(targetFile=os.path.join(output_dir, annotation_filename))