# Data Annotation

## Installation
pip install -r requirements.txt
## Calibration
Subdirectory Calibration include the code for rotate bounding boxes, calibrate height of bboxes and change annotation 
format.
### Usage
1. Prepare path to the input frames as base_dir, path of output calibrated annotation xml file as output_dir, path of 
output calibrated visualization images as vis_dir, path of input annotation xml file as annotation_dir, path of baseline 
annotation xml file as seg_anno_dir.
2. python Rotate_BBOX_Anno.py --base_dir  --output_dir  --vis_dir  --annotation_dir  --seg_anno_dir 
### Implementation
1. Calculate bounding boxes angle Thetha from baseline method bounding boxes.
2. Rotate bounding input bounding boxes by rotating each point theta angle
3. Reshape the height and width of bounding boxes based on the baseline bounding box

## Semi-supervised Annotation
Subdirectory Semi-supervised Annotation include the code for semi-supervised annotation.
### Usage
1. Prepare input_image_path, input_annotation_path, output_annotation_path, output_image_visualization_path as image_path,
annotation_path, save_annotation_path, saved_image_path.
2. python semi_supervised_label.py --image_path --annotation_path --save_annotation_path --saved_image_path --start_point 
--end_point --order
### Implementation
1. Read the start point image with annotation and use the annotation as src_bbox
2. Find the key points and descriptors via SIFT for both source image and destination image.
3. Compute matches for source and destination key points and store the good matches.
4. Calculate the homography matrix via the good matches points.
5. Compute the destination bounding box via perspective transform of source bounding box and homography matrix.
6. Set the destination image and bounding box as source image and bounding box and repeat label the next image until the
end point is reached. 
