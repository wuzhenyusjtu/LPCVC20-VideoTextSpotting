from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import nltk
import cv2
from shapely.geometry import Polygon

# constants
WINDOW_NAME = "COCO detections"
CTLABELS = [' ','!','"','#','$','%','&','\'','(',')','*','+',',','-','.','/','0','1','2','3','4','5','6','7','8','9',':',';','<','=','>','?','@','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','[','\\',']','^','_','`','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','{','|','}','~']

def area(boxes):
    """Computes area of boxes.
    Args:
      boxes: Numpy array with shape [N, 8] holding N boxes
    Returns:
      a numpy array with shape [N*1] representing box areas
    """
    N = boxes.shape[0]
    res = np.zeros((N))
    for n in range(N):
        coord1 = [(boxes[n][2*i], boxes[n][2*i+1]) for i in range(int(len(boxes[n])/2)) ]
        res[n] = Polygon(coord1).area
    return res


def intersection(boxes1, boxes2):
    """Compute pairwise intersection areas between boxes.
    Args:
      boxes1: a numpy array with shape [N, 4] holding N boxes
      boxes2: a numpy array with shape [M, 4] holding M boxes
    Returns:
      a numpy array with shape [N*M] representing pairwise intersection area
    """
    N, M = boxes1.shape[0], boxes2.shape[0]
    res = np.zeros((N, M))
    for n in range(N):
        for m in range(M):
            coord1 = [(boxes1[n][2*i], boxes1[n][2*i+1]) for i in range(int(len(boxes1[n])/2)) ]
            coord2 = [(boxes2[m][2*i], boxes2[m][2*i+1]) for i in range(int(len(boxes2[m])/2)) ]
            res[n][m] = Polygon(coord1).intersection(Polygon(coord2)).area
    return res


def iou(boxes1, boxes2):
    """Computes pairwise intersection-over-union between box collections.
    Args:
      boxes1: a numpy array with shape [N, 4] holding N boxes.
      boxes2: a numpy array with shape [M, 4] holding N boxes.
    Returns:
      a numpy array with shape [N, M] representing pairwise iou scores.
    """
    intersect = intersection(boxes1, boxes2)
    area1 = area(boxes1)
    area2 = area(boxes2)
    union = np.expand_dims(area1, axis=1) + np.expand_dims(
        area2, axis=0) - intersect
    return intersect / union


def ioa(boxes1, boxes2):
    """Computes pairwise intersection-over-area between box collections.
    Intersection-over-area (ioa) between two boxes box1 and box2 is defined as
    their intersection area over box2's area. Note that ioa is not symmetric,
    that is, IOA(box1, box2) != IOA(box2, box1).
    Args:
      boxes1: a numpy array with shape [N, 4] holding N boxes.
      boxes2: a numpy array with shape [M, 4] holding N boxes.
    Returns:
      a numpy array with shape [N, M] representing pairwise ioa scores.
    """
    intersect = intersection(boxes1, boxes2)
    areas = np.expand_dims(area(boxes2), axis=0)
    return intersect / areas

def evaluate(img, predictions, labels, annotation_path, final_iou_score, final_iop_score, final_iog_score, final_edit_distance, edit_distance_image, scale_x, scale_y):
    pred_bbox_list = []
    pred_label = []
    output_image = cv2.UMat(img).get()
    for i in range(len(predictions[0]["instances"].pred_boxes)):
        bezier = predictions[0]["instances"].beziers[i].cpu().numpy()
        bezier = np.expand_dims(bezier, axis=0)
        bezier_list = [bezier_pt for bezier_pt in bezier[0]]
        pred_bbox_list.append(bezier_list)
        word_list = predictions[0]["instances"].recs[i].tolist()
        word = ""
        for index in word_list:
            index = int(index)
            if (index == 96):
                break
            if (index >= len(CTLABELS)):
                print("Out of range")
                continue
            word = word + CTLABELS[index]
        word = word.upper()
        pred_label.append(word)
    pred_bbox = np.array([np.array(bbox) for bbox in pred_bbox_list])
    for ann in annotation_path:
        gt_bbox = ann["bezier_pts"]
        gt_bbox = np.array([np.array(gt_bbox)])
        word_list = ann["rec"]
        word = ""
        for index in word_list:
            index = int(index)
            if (index == 96):
                continue
            if (index >= len(CTLABELS)):
                print("Out of range")
                continue
            word = word + CTLABELS[index]
        gt_bbox[:, ::2] = gt_bbox[:, ::2] * scale_x
        gt_bbox[:, 1::2] = gt_bbox[:, 1::2] * scale_y
        word = word.upper()
        try:
            iou_scores = iou(pred_bbox, gt_bbox)
            iop_scores = ioa(gt_bbox, pred_bbox)
            iog_scores = ioa(pred_bbox, gt_bbox)
            idx = np.where(iou_scores == np.amax(iou_scores))[0]
            if (np.amax(iou_scores) < 0.5):
                final_edit_distance.append(len(word))
                continue
            edit_distance = nltk.edit_distance(word, pred_label[idx[0]])
            if (np.isnan(np.amax(iou_scores)) == False):
                final_iou_score.append(np.amax(iou_scores))
            if (np.isnan(np.amax(iop_scores)) == False):
                final_iop_score.append(np.amax(iop_scores))
            if (np.isnan(np.amax(iog_scores)) == False):
                final_iog_score.append(np.amax(iog_scores))
            edit_distance_image.append(edit_distance)
            final_edit_distance.append(edit_distance)
            output_image = cv2.polylines(output_image, [pred_bbox[idx[0]].reshape(8, 2).reshape(-1, 1, 2).astype(int)], True, (255, 0, 0), 1)
            output_image = cv2.putText(output_image, pred_label[idx[0]], (int(round(pred_bbox[idx[0]][0])), int(round(pred_bbox[idx[0]][1]))), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 1)
        except:
            final_edit_distance.append(len(word))
            print("missed")
    return final_iou_score, final_iop_score, final_iog_score, final_edit_distance, output_image