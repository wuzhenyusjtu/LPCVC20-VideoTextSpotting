import os
import cv2
import numpy as np
from scipy.stats import mode, norm

from libs.region import Region
from libs.utils import apply_canny, dsample_image

class TextDetection(object):

    def __init__(self, img, config, direction='both+'):

        ## Read image
        self.img = img
        #  === add to test down sample === start
        # img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
        # === add to test down sample === finish
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.img = rgb_img
        self.h, self.w = img.shape[:2]

        self.direction = direction
        self.config = config
        self.AREA_LIM = config.AREA_LIM
        self.PERIMETER_LIM = config.PERIMETER_LIM
        self.ASPECT_RATIO_LIM = config.ASPECT_RATIO_LIM
        self.OCCUPATION_INTERVAL = config.OCCUPATION_INTERVAL
        self.COMPACTNESS_INTERVAL = config.COMPACTNESS_INTERVAL
        self.SWT_TOTAL_COUNT = config.SWT_TOTAL_COUNT
        self.SWT_STD_LIM = config.SWT_STD_LIM
        self.STROKE_WIDTH_SIZE_RATIO_LIM = config.STROKE_WIDTH_SIZE_RATIO_LIM
        self.STROKE_WIDTH_VARIANCE_RATIO_LIM = config.STROKE_WIDTH_VARIANCE_RATIO_LIM
        self.STEP_LIMIT = config.STEP_LIMIT
        self.KSIZE = config.KSIZE
        self.ITERATION = config.ITERATION
        self.MARGIN = config.MARGIN

        self.final = rgb_img.copy()

        self.height, self.width = self.img.shape[:2]

        self.gray_img = cv2.cvtColor(self.img.copy(), cv2.COLOR_RGB2GRAY)

        self.canny_img = apply_canny(self.img)

        self.sobelX = cv2.Sobel(self.gray_img, cv2.CV_64F, 1, 0, ksize=-1)
        self.sobelY = cv2.Sobel(self.gray_img, cv2.CV_64F, 0, 1, ksize=-1)

        self.stepsX = self.sobelY.astype(int)  ## Steps are inversed!! (x-step -> sobelY)
        self.stepsY = self.sobelX.astype(int)

        self.magnitudes = np.sqrt(self.stepsX * self.stepsX + self.stepsY * self.stepsY)
        self.gradsX = self.stepsX / (self.magnitudes + 1e-10)
        self.gradsY = self.stepsY / (self.magnitudes + 1e-10)

    def get_MSERegions(self, img):
        mser = cv2.MSER_create()
        regions, bboxes = mser.detectRegions(img)
        return regions, bboxes

    def get_stroke_properties(self, stroke_widths):
        if len(stroke_widths) == 0:
            return (0, 0, 0, 0, 0, 0)
        try:
            most_probable_stroke_width = mode(stroke_widths, axis=None)[0][0]
            most_probable_stroke_width_count = mode(stroke_widths, axis=None)[1][0]
        except IndexError:
            most_probable_stroke_width = 0
            most_probable_stroke_width_count = 0
        try:
            mean, std = norm.fit(stroke_widths)
            x_min, x_max = int(min(stroke_widths)), int(max(stroke_widths))
        except ValueError:
            mean, std, x_min, x_max = 0, 0, 0, 0
        return most_probable_stroke_width, most_probable_stroke_width_count, mean, std, x_min, x_max

    def get_strokes(self, xywh):
        x, y, w, h = xywh
        stroke_widths = np.array([[np.Infinity, np.Infinity]])
        for i in range(y, y + h):
            for j in range(x, x + w):
                if self.canny_img[i, j] != 0:
                    gradX = self.gradsX[i, j]
                    gradY = self.gradsY[i, j]

                    prevX, prevY, prevX_opp, prevY_opp, step_size = i, j, i, j, 0

                    if self.direction == "light":
                        go, go_opp = True, False
                    elif self.direction == "dark":
                        go, go_opp = False, True
                    else:
                        go, go_opp = True, True

                    stroke_width = np.Infinity
                    stroke_width_opp = np.Infinity
                    while (go or go_opp) and (step_size < self.STEP_LIMIT):
                        step_size += 1

                        if go:
                            curX = np.int(np.floor(i + gradX * step_size))
                            curY = np.int(np.floor(j + gradY * step_size))
                            if (curX <= y or curY <= x or curX >= y + h or curY >= x + w):
                                go = False
                            if go and ((curX != prevX) or (curY != prevY)):
                                try:
                                    if self.canny_img[curX, curY] != 0:
                                        if np.arccos(gradX * -self.gradsX[curX, curY] + gradY * -self.gradsY[curX, curY]) < np.pi / 2.0:
                                            stroke_width = int(np.sqrt((curX - i) ** 2  + (curY - j) ** 2))
                                            go = False
                                except IndexError:
                                    go = False

                                prevX = curX
                                prevY = curY

                        if go_opp:
                            curX_opp = np.int(np.floor(i - gradX * step_size))
                            curY_opp = np.int(np.floor(j - gradY * step_size))
                            if (curX_opp <= y or curY_opp <= x or curX_opp >= y + h or curY_opp >= x + w):
                                go_opp = False
                            if go_opp and ((curX_opp != prevX_opp) or (curY_opp != prevY_opp)):
                                try:
                                    if self.canny_img[curX_opp, curY_opp] != 0:
                                        if np.arccos(gradX * -self.gradsX[curX_opp, curY_opp] + gradY * -self.gradsY[curX_opp, curY_opp]) < np.pi/2.0:
                                            stroke_width_opp = int(np.sqrt((curX_opp - i) ** 2  + (curY_opp - j) ** 2))
                                            go_opp = False

                                except IndexError:
                                    go_opp = False

                                prevX_opp = curX_opp
                                prevY_opp = curY_opp

                    stroke_widths = np.append(stroke_widths, [(stroke_width, stroke_width_opp)], axis=0)

        stroke_widths_opp = np.delete(stroke_widths[:, 1], np.where(stroke_widths[:, 1] == np.Infinity))
        stroke_widths = np.delete(stroke_widths[:, 0], np.where(stroke_widths[:, 0] == np.Infinity))
        return stroke_widths, stroke_widths_opp

    def detect(self):
        res9 = np.zeros_like(self.img)
        regions, bboxes = self.get_MSERegions(self.gray_img)
        #TODO regions, bboxes = self.get_MSERegions(self.img)

        n_final_regions = 0

        for i, (region, bbox) in enumerate(zip(regions, bboxes)):
            region = Region(region, bbox)

            if region.area < self.w * self.h * self.AREA_LIM:
                continue

            if region.get_perimeter(self.canny_img) < (2 * (self.w + self.h) * self.PERIMETER_LIM):
                continue

            if region.get_aspect_ratio() > self.ASPECT_RATIO_LIM:
                continue

            occupation = region.get_occupation()
            if (occupation < self.OCCUPATION_INTERVAL[0]) or (occupation > self.OCCUPATION_INTERVAL[1]):
                continue

            compactness = region.get_compactness()
            if (compactness < self.COMPACTNESS_INTERVAL[0]) or (compactness > self.COMPACTNESS_INTERVAL[1]):
                continue

            res9 = region.color(res9)
            # x, y, w, h = bbox

            # stroke_widths, stroke_widths_opp = self.get_strokes((x, y, w, h))
            # if self.direction != "both+":
            #     stroke_widths = np.append(stroke_widths, stroke_widths_opp, axis=0)
            #     stroke_width, stroke_width_count, _, std, _, _ = self.get_stroke_properties(stroke_widths)
            # else:
            #     stroke_width, stroke_width_count, _, std, _, _ = self.get_stroke_properties(stroke_widths)
            #     stroke_width_opp, stroke_width_count_opp, _, std_opp, _, _ = self.get_stroke_properties(stroke_widths_opp)
            #     if stroke_width_count_opp > stroke_width_count:        ## Take the stroke_widths with max of counts stroke_width (most probable one)
            #         stroke_widths = stroke_widths_opp
            #         stroke_width = stroke_width_opp
            #         stroke_width_count = stroke_width_count_opp
            #         std = std_opp
            #
            # if len(stroke_widths) < self.SWT_TOTAL_COUNT:
            #     continue
            #
            # if std > self.SWT_STD_LIM:
            #     continue
            #
            # stroke_width_size_ratio = stroke_width / max(region.w, region.h)
            # if stroke_width_size_ratio < self.STROKE_WIDTH_SIZE_RATIO_LIM:
            #     continue
            #
            # stroke_width_variance_ratio = stroke_width / (std * std + 1e-10)
            # if stroke_width_variance_ratio > self.STROKE_WIDTH_VARIANCE_RATIO_LIM:
            #     n_final_regions += 1
            #     res9 = region.color(res9)

        ## Binarize regions
        if np.count_nonzero(res9) > 0:
            binarized = np.zeros_like(self.gray_img)
            rows, cols, _ = np.where(res9 != [0, 0, 0])
            binarized[rows, cols] = 255

            ## Dilate regions and find contours
            kernel = np.zeros((self.KSIZE, self.KSIZE), dtype=np.uint8)
            kernel[(self.KSIZE // 2)] = 1

            res = np.zeros_like(self.gray_img)
            dilated = cv2.dilate(binarized.copy(), kernel, iterations=self.ITERATION)
            _, contours, hierarchies = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            X_min = self.w - 1
            X_max = 0
            Y_min = self.h - 1
            Y_max = 0
            for i, (contour, hierarchy) in enumerate(zip(contours, hierarchies[0])):
                if hierarchy[-1] != -1:
                    continue

                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                X_min = min(min(box[:, 0]), X_min)
                X_max = max(max(box[:, 0]), X_max)
                Y_min = min(min(box[:, 1]), Y_min)
                Y_max = max(max(box[:, 1]), Y_max)
                cv2.drawContours(self.final, [box], 0, (0, 255, 0), 2)
                cv2.drawContours(res, [box], 0, 255, -1)

            return X_min, Y_min, X_max, Y_max, res

        else:
            X_min, Y_min, X_max, Y_max = 0, 0, 0, 0
            return X_min, Y_min, X_max, Y_max, res9