#!python3

import argparse
import glob
import tempfile
import json
import os
import sys
import copy

import cv2
from ocr_lib import OCRLib
import time

from libs.utils import dsample_image

from edit import edit_distance

CONFIG_FILE = os.path.join(os.path.dirname(__file__), 'config.json')


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run OCR on a video"
    )
    parser.add_argument(
        "--input_video", help="Path to the input video", type=str, required=True
    )
    parser.add_argument(
        "--query_file", help="Path of question text file", type=str, required=True
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
    parser.add_argument(
        "--rcg_scr_th", help="Threshold for kept text", default=0.70, type=float
    )
    parser.add_argument(
        "--config_file", type=str, help="OCRLib config json file", required=True
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


def load_result_index(results_path, rcg_scr_th):
    # a translation table to remove the punctuations
    trans_table = dict.fromkeys(map(ord, '!*^&$@,.:;'), None)
    json_files = glob.glob(os.path.join(results_path, "*.json"))
    txt2frm, frm2txt = {}, {}
    for json_file in json_files:
        frame_no = int(os.path.splitext(os.path.basename(json_file))[0])
        with open(json_file, "r") as h_input:
            results = json.load(h_input)

        frm2txt[frame_no] = []
        for result in results:
            #             if result["rcg_scr"] > rcg_scr_th:
            # remove
            #             text = result["rcg_str"].upper()
            text = result.upper()
            text = text.translate(trans_table)
            frm2txt[frame_no].append(text)
            if text in txt2frm:
                txt2frm[text].append(frame_no)
            else:
                txt2frm[text] = [frame_no]

    return txt2frm, frm2txt


def process_video(input_video, cfg_fn, results_path, sampling_rate):
    reader = OCRLib(cfg_fn)
    vidcap = cv2.VideoCapture(input_video)
    assert vidcap.isOpened()

    success = True
    cnt = 0
    while success:
        success, img = vidcap.read()
        cnt += 1

        if not success:
            break

        if (cnt - 1) % sampling_rate != 0:
            continue
        # print("Frame Num:{}".format(cnt))
        ocr_start = time.time()
        frm_output = os.path.join(results_path, "{}.json".format(cnt))

        ori_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        image = copy.deepcopy(ori_image)
        h, w = image.shape[:2]
        if h == 2160 and w == 3840:
            k_size = 4
        elif h == 1080 and w == 1920:
            k_size = 2
        else:
            k_size = 1  # Just for robustness
        # Down sample image
        image = dsample_image(image, k_size)
        results = reader.process_rgb_image(image)
        print(results)
        with open(frm_output, "w") as h_out:
            json.dump(results, h_out)
        ocr_end = time.time()
        print("[INFO] single frame took {:.4f} seconds including writing to disk".format(ocr_end - ocr_start))


def del_conseq_frm(txt2frm, sample_rate):
    new_txt2frm = {}
    for key, value in txt2frm.items():
        list(set(value)).sort()
        t_cnt = 1
        new_value = [value[0]]
        for i in range(len(value) - 1):
            if value[i + 1] - value[i] <= sample_rate:
                t_cnt += 1
            else:
                if t_cnt > 1:
                    if len(new_value) > 0:
                        new_value.pop()
                    new_value.append(value[i - int(t_cnt / 2)])
                    t_cnt = 1
                new_value.append(value[i + 1])
        new_txt2frm[key] = new_value
    return new_txt2frm


def get_frm_res(frms, query, frm2txt):
    outputs = ""
    q_cnt = 0
    if len(frms) == 1:
        for x in frm2txt[frms[0]]:
            if x == query and q_cnt == 0:
                q_cnt += 1
                continue
            outputs += " {}".format(x)
    else:
        for frm in frms:
            for x in frm2txt[frm]:
                if x == query and q_cnt == 0:
                    q_cnt += 1
                    continue
                outputs += " {}".format(x)
    return outputs


def query_video(
        query_file, input_video, results_path, config_file=CONFIG_FILE, sampling_rate=50, rcg_scr_th=0.70
):
    print(input_video)
    # answer should be only answer.txt
    ans_file = 'answers.txt'
    # Process video
    process_video(input_video, config_file, results_path, sampling_rate)

    # Create database lookup
    txt2frm, frm2txt = load_result_index(results_path, rcg_scr_th)
    # print(frm2txt)

    # Get new txt2frm
    new_txt2frm = del_conseq_frm(txt2frm, sampling_rate)

    # Query results
    with open(query_file, "r") as f_query:
        queries = f_query.read().replace(" ", "").replace("\n", "").split(";")

    # remove old answer.txt
    with open(ans_file, "w", newline='') as f:
        for query in queries:
            if query == '':
                continue
            query = query.upper()
            if query not in new_txt2frm:
                target = query
                target_t = ''
                min_d = len(target)
                for key in new_txt2frm.keys():
                    dis = edit_distance(target, key)
                    if dis < min_d:
                        min_d = dis
                        target_t = key
                        # print question only
                if min_d <= 2:
                    frms = new_txt2frm[target_t]
                    outputs = get_frm_res(frms, target_t, frm2txt)
                    q_ans = "{}: {};".format(query, outputs)
                    print(q_ans)
                else:
                    q_ans = "{}:;".format(query)
                    print(q_ans)
            else:
                frms = txt2frm[query]
                outputs = get_frm_res(frms, query, frm2txt)
                q_ans = "{}: {};".format(query, outputs)
                print(q_ans)
            # output save to a actual file
            f.write(q_ans)


def main():
    # Default way provided by FB to run the script for parsing arguments
    # args = parse_args()

    # Only Works for submission
    with tempfile.TemporaryDirectory() as results_path:
        print("OCR Start!")
        start = time.time()
        # query_video(**vars(args))
        query_video(input_video=sys.argv[1], query_file=sys.argv[2], results_path=results_path)
        print('ORC Done!')
        end = time.time()
        print("Total time: {} second ".format(end - start))


if __name__ == "__main__":
    main()
