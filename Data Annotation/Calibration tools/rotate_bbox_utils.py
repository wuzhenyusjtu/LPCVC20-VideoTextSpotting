from math import radians, sin, cos, atan, degrees
import numpy as np


# Rotate one point regarding the original BBox centroid
def rotate_point(x, y, cx, cy, theta):
    x_rot = cos(theta) * (x - cx) - sin(theta) * (y - cy) + cx
    y_rot = sin(theta) * (x - cx) + cos(theta) * (y - cy) + cy
    return x_rot, y_rot


# Rotate the whole BBox thetha angle regarding the original BBox centroid
def rotate_bbox(bbox, theta):
    x1, y1, x2, y2, x3, y3, x4, y4 = bbox
    cx = (x1 + x3)//2
    cy = (y1 + y2)//2
    x1_rot, y1_rot = rotate_point(x1, y1, cx, cy, theta)
    x2_rot, y2_rot = rotate_point(x2, y2, cx, cy, theta)
    x3_rot, y3_rot = rotate_point(x3, y3, cx, cy, theta)
    x4_rot, y4_rot = rotate_point(x4, y4, cx, cy, theta)
    return x1_rot, y1_rot, x2_rot, y2_rot, x3_rot, y3_rot, x4_rot, y4_rot


# Shrink the height for the rotated BBox
def shear_bbox(bbox, percent):
    x1, y1, x2, y2, x3, y3, x4, y4 = bbox
    x1_sh = x1 + (x2 - x1) * percent
    y1_sh = y1 + (y2 - y1) * percent
    x2_sh = x2 - (x2 - x1) * percent
    y2_sh = y2 - (y2 - y1) * percent
    x3_sh = x3 - (x3 - x4) * percent
    y3_sh = y3 - (y3 - y4) * percent
    x4_sh = x4 + (x3 - x4) * percent
    y4_sh = y4 + (y3 - y4) * percent
    return x1_sh, y1_sh, x2_sh, y2_sh, x3_sh, y3_sh, x4_sh, y4_sh


# Find the min bounding rectangle for group of points
from scipy.spatial import ConvexHull
def minimum_bounding_rectangle(points):
    """
    Find the smallest bounding rectangle for a set of points.
    Returns a set of points representing the corners of the bounding box.

    :param points: an nx2 matrix of coordinates
    :rval: an nx2 matrix of coordinates
    """
    from scipy.ndimage.interpolation import rotate
    pi2 = np.pi/2.

    # get the convex hull for the points
    hull_points = points[ConvexHull(points).vertices]

    # calculate edge angles
    edges = np.zeros((len(hull_points)-1, 2))
    edges = hull_points[1:] - hull_points[:-1]

    angles = np.zeros((len(edges)))
    angles = np.arctan2(edges[:, 1], edges[:, 0])

    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)

    # find rotation matrices
    rotations = np.vstack([
        np.cos(angles),
        np.cos(angles-pi2),
        np.cos(angles+pi2),
        np.cos(angles)]).T

    rotations = rotations.reshape((-1, 2, 2))

    # apply rotations to the hull
    rot_points = np.dot(rotations, hull_points.T)

    # find the bounding points
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

    # find the box with the best area
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)

    # return the best box
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]

    rval = np.zeros((4, 2))
    rval[0] = np.dot([x1, y2], r)
    rval[1] = np.dot([x2, y2], r)
    rval[2] = np.dot([x2, y1], r)
    rval[3] = np.dot([x1, y1], r)

    return rval


# Compute angle for BBox
# Assume the BBox matches well with the text
def bbox_angle(bbox):
    x1, y1, x2, y2, x3, y3, x4, y4 = bbox
    slope = (y4 - y1)/(x4 - x1)
    radian = atan(slope)
    angle = degrees(radian)
    return angle


# This for backup only, when masktextspotter bbox has no element
def detect_angle(image, xmin, ymin, xmax, ymax):
    w = (xmax - xmin)
    h = (ymax - ymin)

    image = image[ymin:ymin + h, xmin:xmin + w]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)

    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    thetha = radians(angle)
    return thetha