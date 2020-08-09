import os
import cv2
import numpy as np
from PyQt5.QtGui import QImage
from math import radians
from read_write_xml import PascalVocWriter, parse_rec
from rotate_bbox_utils import minimum_bounding_rectangle, bbox_angle, rotate_bbox, detect_angle, shear_bbox


def main():
    cnt = 0
    base_dir = "../Rotate_Anno_Test_Data/oblique_img_modified_anno/oblique_images/"
    output_dir = "../Rotate_Anno_Test_Data/oblique_img_modified_anno/modified_anno/"
    vis_dir = "../Rotate_Anno_Test_Data/oblique_img_modified_anno/modified_anno_visualization/"
    annotation_dir = "../Rotate_Anno_Test_Data/oblique_img_modified_anno/original_anno/"
    seg_anno_dir = "../Rotate_Anno_Test_Data/oblique_img_modified_anno/masktextspotter_anno/"
    for filename in os.listdir(base_dir):
        # counter
        cnt = cnt + 1
        print(cnt)

        image = cv2.imread(os.path.join(base_dir, filename))
        image1 = np.copy(image)
        image2 = np.copy(image)
        image3 = np.copy(image)
        annotation_filename = filename.replace(".jpg", ".xml")
        seg_anno_filename = filename.replace(".jpg", ".txt")

        if os.path.exists(os.path.join(annotation_dir, annotation_filename)):
            annotation_path = os.path.join(annotation_dir, annotation_filename)
            seg_anno_path = os.path.join(seg_anno_dir, seg_anno_filename)
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

            # Parse the segmentation points generated by MaskTextSpotter
            file = open(seg_anno_path, 'r')
            seg_anno = eval(file.read())
            file.close()

            # Parse the original bbox, and initiaize
            bboxes = parse_rec(annotation_path)

            # Compute the angle for the text in the image
            Thetha = []
            # bb2 is the reference bounding box generate from MaskTextSpotter
            bb2 = []
            bb1 = []  # Original annotation bbox

            if len(seg_anno) > 0:
                for i in range(0, len(seg_anno)):
                    pts = np.array(seg_anno[i], np.int32)
                    pts = pts.reshape((-1, 2))
                    min_bbox = minimum_bounding_rectangle(pts)
                    [x1, y1], [x2, y2], [x3, y3], [x4, y4] = min_bbox
                    # coordinate orientation different from ours, need rearrange
                    if x4 < x2:
                        bb2.append([x3, y3, x4, y4, x1, y1, x2, y2])
                    else:
                        bb2.append([x2, y2, x3, y3, x4, y4, x1, y1])
                    angle = bbox_angle(bb2[i])
                    radian = radians(angle)
                    Thetha.append(radian)
                # Remove max and min in Thetha to avoid extreme cases
                if len(Thetha) > 3:
                    Thetha.remove(max(Thetha))
                    Thetha.remove(min(Thetha))

                for bbox in bboxes:
                    bbox_coordinates = bbox['bbox']
                    x_min = bbox_coordinates[0]
                    y_min = bbox_coordinates[1]
                    x_max = bbox_coordinates[2]
                    y_max = bbox_coordinates[3]
                    bb1.append([x_min, y_min, x_min, y_max, x_max, y_max, x_max, y_min])
            else:
                Thetha = []

                for bbox in bboxes:
                    bbox_coordinates = bbox['bbox']
                    x_min = bbox_coordinates[0]
                    y_min = bbox_coordinates[1]
                    x_max = bbox_coordinates[2]
                    y_max = bbox_coordinates[3]
                    bb1.append([x_min, y_min, x_min, y_max, x_max, y_max, x_max, y_min])
                    thetha = detect_angle(image, x_min, y_min, x_max, y_max)
                    Thetha.append(thetha)

            thetha = sum(Thetha) / len(Thetha)

            # Rotate Original bounding box and store the new coordinate in bb3
            bb3 = []
            for i in range(0, len(bb1)):
                x1, y1, x2, y2, x3, y3, x4, y4 = rotate_bbox(bb1[i], thetha)
                if x1 < 0:
                    x1 = 0
                if y1 < 0:
                    y1 = 0
                if x2 < 0:
                    x2 = 0
                if y2 > image.shape[0]:
                    y2 = image.shape[0]
                if x3 > image.shape[1]:
                    x3 = image.shape[1]
                if y3 > image.shape[0]:
                    y3 = image.shape[0]
                if x4 > image.shape[1]:
                    x4 = image.shape[1]
                if y4 < 0:
                    y4 = 0
                bb3.append([x1, y1, x2, y2, x3, y3, x4, y4])

            from math import sqrt
            w = []
            # When len(bb2)>0, matched bb2 with the order of bb3
            if len(bb2) > 0:
                #     bb4 = match_bboxes(bb3, bb2)
                # Compute average height for bounding boxes generated by MaskTextSpotter
                for i in range(0, len(bb2)):
                    x1, y1, x2, y2, x3, y3, x4, y4 = bb2[i]
                    w.append(sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))
            else:
                #     bb4 = bb3.copy()
                # Compute average height for original bounding boxes
                for i in range(0, len(bb3)):
                    x1, y1, x2, y2, x3, y3, x4, y4 = bb3[i]
                    w.append(sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))
            width = sum(w) / len(w)

            # bb4 is the final bbox
            # reshape the rotated bbox, assume MaskTextSpotter bbox hieght is within rotated bbox
            bb4 = []
            for i in range(0, len(bb3)):
                x1, y1, x2, y2, x3, y3, x4, y4 = bb3[i]
                w = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                if w == 0:
                    percent = 0
                else:
                    percent = abs((w - width) / (2 * w) * 0.8)
                bb4.append(shear_bbox(bb3[i], percent))

            for i in range(0, len(bb1)):
                x1, y1, x2, y2, x3, y3, x4, y4 = bb1[i]
                pts = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.int32).reshape((-1, 1, 2))
                image_m = cv2.polylines(image3, [pts], True, (255, 0, 0), 2)
            if len(bb2) > 0:
                for i in range(0, len(bb2)):
                    x1, y1, x2, y2, x3, y3, x4, y4 = bb2[i]
                    pts = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.int32).reshape((-1, 1, 2))
                    image_mts = cv2.polylines(image, [pts], True, (255, 0, 0), 2)
            else:
                image_mts = image.copy()
            for i in range(0, len(bb3)):
                x1, y1, x2, y2, x3, y3, x4, y4 = bb3[i]
                pts = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.int32).reshape((-1, 1, 2))
                image_r = cv2.polylines(image1, [pts], True, (255, 0, 0), 2)

            for i in range(0, len(bb4)):
                label = bboxes[i]['name']
                difficult = bboxes[i]['difficult']
                x1, y1, x2, y2, x3, y3, x4, y4 = bb4[i]
                writer.addBndBox(x1, y1, x2, y2, x3, y3, x4, y4, label, difficult)
                pts = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.int32).reshape((-1, 1, 2))
                image_mr = cv2.polylines(image2, [pts], True, (255, 0, 0), 2)

            # Save test images
            image_1 = np.hstack([image_mr, image_mts])
            image_2 = np.hstack([image_r, image_m])
            image_vis = np.vstack([image_1, image_2])
            cv2.imwrite(os.path.join(vis_dir, filename), image_vis)
            writer.save(targetFile=os.path.join(output_dir, annotation_filename))


if __name__ == '__main__':
    main()
