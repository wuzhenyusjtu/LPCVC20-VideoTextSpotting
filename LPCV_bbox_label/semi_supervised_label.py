from Pascal import *
from PIL import Image
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run semi automatic label algorithm"
    )
    parser.add_argument(
        "--image_path", help="Path to the input images", type=str, required=True
    )
    parser.add_argument(
        "--annotation_path", help="Path of preselected annotation files", type=str, required=True
    )
    parser.add_argument(
        "--saved_image_path", help="Path to the modified input images", type=str, required=True
    )
    parser.add_argument(
        "--saved_annotation_path", help="Path of generated annotation files", type=str, required=True
    )
    parser.add_argument(
        "--start_points_list", help="Image index where semi auto label algorithm starts", type=list, required=True
    )
    parser.add_argument(
        "--end_points_list", help="Image index where semi auto label algorithm ends", typep=list, required=True
    )
    parser.add_argument(
        "--order_list", help="Flag for forward or reverse when the algorithm starts", typep=list, required=True
    )

    args = parser.parse_args()

    assert len(args.start_points_list) != len(args.end_points_list), "Start_points_list and end_points_list not " \
                                                                     "matched. "
    if len(args.order_list) != len(args.start_points_list):
        args.order_list = [1]*len(args.start_points_list)
    return args


def main():
    # Initialize input
    manual_annotated = []
    args = parse_args()
    start_point = args.start_points_list
    end_point = args.end_points_list
    order = args.order_list
    image_path = args.image_path
    annotation_path = args.annotation_path
    save_annotation_path = args.saved_annotation_path
    saved_image_path = args.saved_image_path

    for i in range(0, len(start_point)):
        start_pt = start_point[i]
        end_pt = end_point[i]
        flag_o = order[i]
        if order[i] == 1:
            idx = start_pt
        else:
            idx = end_pt
        base_dir = os.path.basename(image_path)
        src_image = cv2.imread(os.path.join(image_path, base_dir + '_' + str(idx) + '.jpg'))
        src_bboxes = parse_rec(os.path.join(annotation_path, base_dir + '_' + str(idx) + '.xml'))
        [ans_image, ans_bboxes, flag1] = overlay_bbox_image(src_image, src_image, src_bboxes, False, idx, flag_o)

        count = 0
        while count < (end_pt - start_pt):
            count = count + 1
            if order[i] == 1:
                idx = idx + 1
            else:
                idx = idx - 1
            dst_imagename = base_dir + '_' + str(idx) + '.jpg'
            dst_annotation_filename = base_dir + '_' + str(idx) + '.xml'
            dst_image = cv2.imread(os.path.join(image_path, dst_imagename))

            [dst_image, dst_bboxes, flag1] = overlay_bbox_image(src_image, dst_image, src_bboxes, True, idx, flag_o)
            print(idx)
            if flag1:
                print(os.path.join(image_path, dst_imagename))
                manual_annotated.append(idx)
                continue
            else:
                cv2.imwrite(os.path.join(saved_image_path, str(idx) + '.jpg'), dst_image)

            imagePath = os.path.join(image_path, dst_imagename)
            imgFolderPath = os.path.dirname(imagePath)
            imgFolderName = os.path.split(imgFolderPath)[-1]
            imgFileName = os.path.basename(imagePath)
            image = Image.open(os.path.join(image_path, dst_imagename))
            width, height = image.size
            imageShape = [height, width, 1 if np.array(image).shape[2] == 1 else 3]
            writer = PascalVocWriter(imgFolderName, imgFileName, imageShape, localImgPath=imagePath)
            writer.verified = False

            for obj in dst_bboxes:
                label = obj['name']
                difficult = obj['difficult']
                obj['bbox'][0] = max(0, obj['bbox'][0])
                obj['bbox'][0] = min(obj['bbox'][0], width-1)
                obj['bbox'][2] = max(0, obj['bbox'][2])
                obj['bbox'][2] = min(obj['bbox'][2], width-1)
                obj['bbox'][1] = max(0, obj['bbox'][1])
                obj['bbox'][1] = min(obj['bbox'][1], height-1)
                obj['bbox'][3] = max(0, obj['bbox'][3])
                obj['bbox'][3] = min(obj['bbox'][3], height-1)
                writer.addBndBox(obj['bbox'][0], obj['bbox'][1], obj['bbox'][2], obj['bbox'][3], label, difficult)

            writer.save(targetFile=os.path.join(save_annotation_path, dst_annotation_filename))
            src_image = dst_image
            src_bboxes = dst_bboxes


if __name__ == '__main__':
    main()
