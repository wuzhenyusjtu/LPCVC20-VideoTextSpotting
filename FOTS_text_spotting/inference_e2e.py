import numpy as np
import torch
import os
import json
import cv2
import argparse
import multiprocessing as mp

from detectron2.utils.logger import setup_logger
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine.defaults import DefaultPredictor

from fots_recognizer.config import get_cfg

from eval_e2e import evaluate

class DefaultPredictor:
    """
    Create a simple end-to-end predictor with the given config that runs on
    single device for a single input image.
    Compared to using the model directly, this class does the following additions:
    1. Load checkpoint from `cfg.MODEL.WEIGHTS`.
    2. Always take BGR image as the input and apply conversion defined by `cfg.INPUT.FORMAT`.
    3. Apply resizing defined by `cfg.INPUT.{MIN,MAX}_SIZE_TEST`.
    4. Take one input image and produce a single output, instead of a batch.
    If you'd like to do anything more fancy, please refer to its source code
    as examples to build and use the model manually.
    Attributes:
        metadata (Metadata): the metadata of the underlying dataset, obtained from
            cfg.DATASETS.TEST.
    Examples:
    ::
        pred = DefaultPredictor(cfg)
        inputs = cv2.imread("input.jpg")
        outputs = pred(inputs)
    """

    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

    def __call__(self, original_image):
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            height, width = original_image.shape[2:]
            image = original_image
            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])[0]
            return predictions

class VisualizationDemo(object):
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.cpu_device = torch.device("cpu")
        self.predictor = DefaultPredictor(cfg)

    def run_on_image(self, image):
        predictions = self.predictor(image)
        return predictions

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    print(args.config_file)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 Demo")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/e2e_mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser



if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    phase = "test"
    ann_file = phase + "_16pt"

    base_dir = "path_to_dataset_dir"
    out_dir = "path_to_save_predictions"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    image_dir = os.path.join(base_dir, "test_images")
    annotation_dir = os.path.join(base_dir, ann_file  +".json")
    f = open(annotation_dir, )
    data = json.load(f)
    image_dict = {}
    ann_dict = {}
    image2ann_dict = {}
    cnt = 0
    for image in data["images"]:
        cnt = cnt + 1
        image_dict[image["id"]] = image
    cnt = 0
    for ann in data["annotations"]:
        cnt = cnt + 1
        ann_dict[ann["id"]] = ann
        if ann["image_id"] not in image2ann_dict:
            image2ann_dict[ann["image_id"]] = []
        image2ann_dict[ann["image_id"]].append(ann["id"])
    final_iou_score = []
    final_iop_score = []
    final_iog_score = []
    final_edit_distance = []
    final_time_taken = []
    final_timestamp = []
    edit_distance_image_map = {}
    cnt = 0
    for image_id in image2ann_dict:
        filename = image_dict[image_id]["file_name"]
        ann_list = []
        for ann_id in image2ann_dict[image_id]:
            ann_list.append(ann_dict[ann_id])
        # use PIL, to be consistent with evaluation
        cnt = cnt + 1
        print(cnt)
        image_path = os.path.join(image_dir, filename)
        print(image_path)
        import time
        start_time = time.time()
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        scale_x = 480 / image.shape[1]  # 2240 # 1280
        scale_y = 288 / image.shape[0]  # 1248 # 704
        scaled_image = cv2.resize(image, dsize=(0, 0), fx=scale_x, fy=scale_y, interpolation=cv2.INTER_CUBIC)
        orig_scaled_image = scaled_image.copy()

        scaled_image = scaled_image[:, :, ::-1].astype(np.float32)
        scaled_image = (scaled_image / 255 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        image_tensor = torch.from_numpy(np.expand_dims(np.transpose(scaled_image, axes=(2, 0, 1)), axis=0)).float()
        predictions = demo.run_on_image(image_tensor)
        annotation_file = filename.replace(".jpg", ".xml")
        annotation_path = os.path.join(annotation_dir, annotation_file)
        edit_distance_image = []
        [final_iou_score, final_iop_score, final_iog_score, final_edit_distance, output_image] = evaluate(orig_scaled_image, predictions[0], predictions[1], ann_list, final_iou_score, final_iop_score, final_iog_score, final_edit_distance, edit_distance_image, scale_x, scale_y)
        try:
            edt_img = sum(edit_distance_image) / len(edit_distance_image)
            edit_distance_image_map[filename] = edt_img
        except:
            edit_distance_image_map[filename] = "no_detection"
        cv2.imwrite(os.path.join(out_dir, filename), output_image)
        logger.info(
            "{}: detected {} instances in {:.2f}s".format(
                image_path, len(predictions[0][0]["instances"]), time.time() - start_time
            )
        )
        if (cnt % 100 == 0):
            try:
                print('IOU' + str(sum(final_iou_score) / len(final_iou_score)))
                print('IOP' + str(sum(final_iop_score) / len(final_iop_score)))
                print('IOG' + str(sum(final_iog_score) / len(final_iog_score)))
                print('Edit Distance' + str(sum(final_edit_distance) / len(final_edit_distance)))
                print('Inference Time Taken: ' + str(sum(final_time_taken) / len(final_time_taken)))
            except:
                print('Edit Distance' + str(sum(final_edit_distance) / len(final_edit_distance)))
    try:
        print('IOU' + str(sum(final_iou_score) / len(final_iou_score)))
        print('IOP' + str(sum(final_iop_score) / len(final_iop_score)))
        print('IOG' + str(sum(final_iog_score) / len(final_iog_score)))
        print('Edit Distance' + str(sum(final_edit_distance) / len(final_edit_distance)))
        print('Inference Time Taken: ' + str(sum(final_time_taken) / len(final_time_taken)))
    except:
        print('Edit Distance' + str(sum(final_edit_distance) / len(final_edit_distance)))