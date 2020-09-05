from Pascal import *
from PIL import Image
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run semi supervised annotation on frames"
    )
    parser.add_argument(
        "--image_path", help="Path to the input frames", type=str, required=True
    )
    parser.add_argument(
        "--annotation_path", help="Path of input manual labeled annotation xml file", type=str, required=True
    )
    parser.add_argument(
        "--save_annotation_path", help="Path of output auto labeled annotation xml file", type=str, required=True
    )
    parser.add_argument(
        "--saved_image_path", help="Path of output auto labeled images visualization", type=str, required=True
    )
    parser.add_argument(
        "--start_point", help="List of manual labeled frames number", type=list, required=True
    )
    parser.add_argument(
        "--end_point", help="List of end frames number that matches the start_point", type=list, required=True
    )
    parser.add_argument(
        "--order", help="list for label order, 0 for reverse order, 1 for normal order", type=list, required=True
    )

    args = parser.parse_args()

    if len(args.start_point) != len(args.end_point) or len(args.order) != len(args.start_point):
        raise ValueError('Length of start_point must equal length of end_point and order')

    return args


def main():
    # Initialize input
    manual_annotated = []
    args = parse_args()

    # Example for arms_1st
    start_point = args.start_point
    end_point = args.end_point
    order = args.order
    image_path = args.image_path
    annotation_path = args.annotation_path
    save_annotation_path = args.save_annotation_path
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
