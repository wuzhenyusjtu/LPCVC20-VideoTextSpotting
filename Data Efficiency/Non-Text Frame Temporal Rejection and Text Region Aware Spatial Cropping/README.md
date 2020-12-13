# Non-Text Frame Temporal Rejection and Text Region Aware Spatial Cropping
Directory includes codes for improving data efficiency based on traditional computer vision methods. The codes includes non-text frame temporal rejection and text region aware spatial cropping.
## Installation
pip install -r requirements.txt
## Usage
python main.py --input_video input_video_path
### Implementation
1. Downsample input image.
2. Convert the down-sampled image to YUV format.
3. Apply canny edge method for all YUV format images.
4. Sum up pixels for canny images converted from YUV format images.
5. Apply closing to the sum up image.
6. Sum up pixel values among x and y directory for the closing image.
7. Find peaks and mean values for the sum up values for both x-axis and y-axis.
8. Reject images with low mean pixel values (preset threshold) or 
   low peaks number (preset threshold, default <=2).
9. Skip images with extremely high mean pixels values (Ignore images with complicated background).
10. Crop images based on peaks (default pick peak[1] and peak[len(peak)-2]).
