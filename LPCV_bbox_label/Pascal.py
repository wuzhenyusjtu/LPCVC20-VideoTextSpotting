import os
import cv2
import numpy as np
from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement
from lxml import etree
import codecs
from xml_libs.constants import DEFAULT_ENCODING
from xml_libs.ustr import ustr

XML_EXT = '.xml'
ENCODE_METHOD = DEFAULT_ENCODING


class PascalVocWriter:

    def __init__(self, foldername, filename, imgSize, databaseSrc='Unknown', localImgPath=None):
        self.foldername = foldername
        self.filename = filename
        self.databaseSrc = databaseSrc
        self.imgSize = imgSize
        self.boxlist = []
        self.localImgPath = localImgPath
        self.verified = False

    def prettify(self, elem):
        """
            Return a pretty-printed XML string for the Element.
        """
        rough_string = ElementTree.tostring(elem, 'utf8')
        root = etree.fromstring(rough_string)
        return etree.tostring(root, pretty_print=True, encoding=ENCODE_METHOD).replace("  ".encode(), "\t".encode())
        # minidom does not support UTF-8

    def genXML(self):
        """
            Return XML root
        """
        # Check conditions
        if self.filename is None or \
                self.foldername is None or \
                self.imgSize is None:
            return None

        top = Element('annotation')
        if self.verified:
            top.set('verified', 'yes')

        folder = SubElement(top, 'folder')
        folder.text = self.foldername

        filename = SubElement(top, 'filename')
        filename.text = self.filename

        if self.localImgPath is not None:
            localImgPath = SubElement(top, 'path')
            localImgPath.text = self.localImgPath

        source = SubElement(top, 'source')
        database = SubElement(source, 'database')
        database.text = self.databaseSrc

        size_part = SubElement(top, 'size')
        width = SubElement(size_part, 'width')
        height = SubElement(size_part, 'height')
        depth = SubElement(size_part, 'depth')
        width.text = str(self.imgSize[1])
        height.text = str(self.imgSize[0])
        if len(self.imgSize) == 3:
            depth.text = str(self.imgSize[2])
        else:
            depth.text = '1'

        segmented = SubElement(top, 'segmented')
        segmented.text = '0'
        return top

    def addBndBox(self, xmin, ymin, xmax, ymax, name, difficult):
        bndbox = {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax, 'name': name, 'difficult': difficult}
        self.boxlist.append(bndbox)

    def appendObjects(self, top):
        for each_object in self.boxlist:
            object_item = SubElement(top, 'object')
            name = SubElement(object_item, 'name')
            name.text = ustr(each_object['name'])
            pose = SubElement(object_item, 'pose')
            pose.text = "Unspecified"
            gentype = SubElement(object_item, 'gentype')
            gentype.text = "auto"
            truncated = SubElement(object_item, 'truncated')
            if int(float(each_object['ymax'])) == int(float(self.imgSize[0])) or (int(float(each_object['ymin'])) == 1):
                truncated.text = "1"  # max == height or min
            elif (int(float(each_object['xmax'])) == int(float(self.imgSize[1]))) or (
                    int(float(each_object['xmin'])) == 1):
                truncated.text = "1"  # max == width or min
            else:
                truncated.text = "0"
            difficult = SubElement(object_item, 'difficult')
            difficult.text = str(bool(each_object['difficult']) & 1)
            bndbox = SubElement(object_item, 'bndbox')
            xmin = SubElement(bndbox, 'xmin')
            xmin.text = str(each_object['xmin'])
            ymin = SubElement(bndbox, 'ymin')
            ymin.text = str(each_object['ymin'])
            xmax = SubElement(bndbox, 'xmax')
            xmax.text = str(each_object['xmax'])
            ymax = SubElement(bndbox, 'ymax')
            ymax.text = str(each_object['ymax'])

    def save(self, targetFile=None):
        root = self.genXML()
        self.appendObjects(root)
        out_file = None
        if targetFile is None:
            out_file = codecs.open(
                self.filename + XML_EXT, 'w', encoding=ENCODE_METHOD)
        else:
            out_file = codecs.open(targetFile, 'w', encoding=ENCODE_METHOD)

        prettifyResult = self.prettify(root)
        out_file.write(prettifyResult.decode('utf8'))
        out_file.close()


def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ElementTree.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        #         obj_struct['gentype'] = obj.find('gentype').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)

    return objects


def matching_points(image1, image2, flag, roi_coordinate):
    im1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    im2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    mask = np.zeros(im1.shape[:2], dtype=np.uint8)

    # draw your selected ROI on the mask image
    cv2.rectangle(mask, (int(round(roi_coordinate[0])), int(round(roi_coordinate[1]))),
                  (int(round(roi_coordinate[2])), int(round(roi_coordinate[3]))), 255, thickness=-1)

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the key points and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(im1, mask)
    kp2, des2 = sift.detectAndCompute(im2, mask)

    if flag == 0:
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

        matches = bf.knnMatch(des1, des2, k=2)

    else:
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    src_points = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_points = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    return src_points, dst_points, kp1, kp2, good


def overlay_bbox_image(src_image, dst_image, bboxes, flag, idx, order):
    bbox_temp = []
    flag1 = False
    if order == 1:
        prev_idx = idx - 1
    else:
        prev_idx = idx + 1

    if flag:
        [x_min, y_min, x_max, y_max] = bboxes[0]['bbox']
        # Get the Region of Interest based on bounding boxes
        for bbox in bboxes:
            bbox_coordinates = bbox['bbox']
            x_min = min(x_min, bbox_coordinates[0])
            y_min = min(y_min, bbox_coordinates[1])
            x_max = max(x_max, bbox_coordinates[2])
            y_max = max(y_max, bbox_coordinates[3])
        # Enlarge the ROI region
        x_min = max(x_min - 50, 0)
        y_min = max(y_min - 100, 0)
        x_max = min(x_max + 50, src_image.shape[1] - 1)
        y_max = min(y_max + 100, src_image.shape[0] - 1)
        roi = [x_min, y_min, x_max, y_max]

        for bbox in bboxes:
            bbox_coordinates = bbox['bbox']
            bbox_coordinate = np.array(
                [[[bbox_coordinates[0], bbox_coordinates[1]]], [[bbox_coordinates[2], bbox_coordinates[1]]],
                 [[bbox_coordinates[0], bbox_coordinates[3]]], [[bbox_coordinates[2], bbox_coordinates[3]]]],
                dtype=np.float32)

            [src_points, dst_points, kp1, kp2, good] = matching_points(src_image, dst_image, 0, roi)
            feature_matching_image = cv2.drawMatches(src_image, kp1, dst_image, kp2, good, None, flags=2)
            # Save the feature matching images
            cv2.imwrite(os.path.join('Sample_Dataset_Annotated/bidc_base/Semi-Supervised_Feature_Matching/',
                                     str(prev_idx) + '+' +
                                     str(idx) + '.jpg'), feature_matching_image)

            if src_points.shape[0] < 4:
                print("hi")
                flag1 = True
                return dst_image, bbox_temp, flag1

            projective_matrix, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
            bbox_coordinate = cv2.perspectiveTransform(bbox_coordinate, projective_matrix)

            bbox_temp.append(bbox)
            bbox_temp[-1]['bbox'][0] = int(round(bbox_coordinate[0][0][0]))
            bbox_temp[-1]['bbox'][1] = int(round(bbox_coordinate[0][0][1]))
            bbox_temp[-1]['bbox'][2] = int(round(bbox_coordinate[3][0][0]))
            bbox_temp[-1]['bbox'][3] = int(round(bbox_coordinate[3][0][1]))

            start_point = (int(round(bbox_coordinate[0][0][0])), int(round(bbox_coordinate[0][0][1])))
            end_point = (int(round(bbox_coordinate[3][0][0])), int(round(bbox_coordinate[3][0][1])))
            dst_image = cv2.rectangle(dst_image, start_point, end_point, (255, 0, 0), 2)
    else:
        for bbox in bboxes:
            bbox_temp.append(bbox)
            bbox_coordinates = bbox['bbox']
            start_point = (bbox_coordinates[0], bbox_coordinates[1])
            end_point = (bbox_coordinates[2], bbox_coordinates[3])
            dst_image = cv2.rectangle(dst_image, start_point, end_point, (255, 0, 0), 2)
    return dst_image, bbox_temp, flag1
