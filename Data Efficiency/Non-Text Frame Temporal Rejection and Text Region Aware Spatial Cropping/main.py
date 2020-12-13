#!python3

import argparse
import os
import cv2
import numpy as np
import time
from scipy.signal import find_peaks


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run OCR on a video"
    )
    parser.add_argument(
        "--input_video", help="Path to the input video", type=str, required=True
    )
    parser.add_argument(
        "--results_path",
        help="Path to store the results from video",
        default=None,
        type=str,
    )

    parser.add_argument(
        "--sampling_rate", help="Sample every x frames", default=30, type=int
    )

    args = parser.parse_args()

    if args.results_path is None:
        args.results_path = (
                os.path.splitext(os.path.basename(args.input_video))[0] + "_results"
        )
        print('Using "{}" as results cache path'.format(args.results_path))
        if not os.path.exists(args.results_path):
            os.makedirs(args.results_path)

    return args


def dsample_image(img, ksize):
    h, w = img.shape[:2]
    resized_img = np.lib.stride_tricks.as_strided(
        img,
        shape=(int(h / ksize), int(w / ksize), ksize, ksize, 3),
        strides=img.itemsize * np.array([ksize * w * 3, ksize * 3, w * 3, 1 * 3, 1]))
    return resized_img[:, :, 0, 0].copy()


def apply_canny(img, sigma=0.33):
    v = np.median(img)
    lower = int(max(0, (1.0-sigma)*v))
    upper = int(min(255, (1.0+sigma)*v))
    return cv2.Canny(img, lower, upper)


def skip_images(mean_x, mean_y):
    if mean_x > 12000 and mean_y > 15000:
        return True
    else:
        return False


def reject(peaks_x, peaks_y, mean_x, mean_y):
    if (mean_x < 400 and mean_y < 400) or len(peaks_x) <= 1 or len(peaks_y) <= 1:
        return False
    else:
        return True


def select(input_image):
    yuv_img = cv2.cvtColor(input_image, cv2.COLOR_BGR2YUV)
    canny_gray = apply_canny(yuv_img[:, :, 0])
    canny_U = apply_canny(yuv_img[:, :, 1])
    canny_V = apply_canny(yuv_img[:, :, 2])
    auto = canny_U | canny_V | canny_gray

    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(auto, cv2.MORPH_CLOSE, kernel)
    hist_x = np.sum(closing, axis=0)
    hist_y = np.sum(closing, axis=1)
    mean_x = np.mean(hist_x)
    mean_y = np.mean(hist_y)
    peaks_x, _ = find_peaks(hist_x)
    peaks_y, _ = find_peaks(hist_y)

    # First stage rejecter and cropping + second stage rejecter
    if reject(peaks_x, peaks_y, mean_x, mean_y):

        if skip_images(mean_x, mean_y):
            xmin, xmax, ymin, ymax = int(peaks_x[0]), int(peaks_x[-1]), int(peaks_y[0]), int(peaks_y[-1])
            return input_image[ymin:ymax, xmin:xmax]
        else:
            xmin, xmax, ymin, ymax = int(peaks_x[0]), int(peaks_x[-1]), int(peaks_y[0]), int(peaks_y[-1])

        if xmin > 15:
            xmin -= 15
        if xmax < len(hist_x) - 15:
            xmax += 15
        if ymin > 15:
            ymin -= 15
        if ymax < len(hist_y) - 15:
            ymax += 15
        return input_image[ymin:ymax, xmin:xmax]

    # Rejected by first stage rejecter
    else:
        return np.array([])


def main():
    # Default way provided by FB to run the script for parsing arguments
    args = parse_args()
    vidcap = cv2.VideoCapture(args.input_video)
    assert vidcap.isOpened()

    success = True
    cnt = 0

    sampling_rate = args.sampling_rate
    while success:
        success, img = vidcap.read()
        cnt += 1

        if not success:
            break

        if (cnt - 1) % sampling_rate != 0:
            continue
        # print("Frame Num:{}".format(cnt))
        ocr_start = time.time()
        frm_output = os.path.join(args.results_path, "{}.jpg".format(cnt))
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w = image.shape[:2]
        if h == 2160 and w == 3840:
            k_size = 4
        elif h == 1080 and w == 1920:
            k_size = 2
        else:
            k_size = 1  # Just for robustness
        # Down sample image
        image = dsample_image(image, k_size)
        image = select(image)
        ocr_end = time.time()
        print("[INFO] single frame took {:.4f} seconds to select".format(ocr_end - ocr_start))
        cv2.imwrite(frm_output, image)


if __name__ == "__main__":
    main()
