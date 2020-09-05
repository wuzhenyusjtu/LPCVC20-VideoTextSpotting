# Non-Text Frame Temporal Rejection and Text Region Aware Spatial Cropping
Directory includes codes for solution for LPCV 2020 competition, which includes non-text frame temporal rejection and text region
aware spatial cropping, OCR detection based on EAST model and recognition based on CRNN model.
## Installation
pip install -r requirements.txt
## Usage
python main.py input_video_path query_file_path
### Implementation
1. Convert image to YUV format.
2. Apply canny edge method for all Y U V format images.
3. Sum up pixels for all canny images.
4. Apply closing to the sum up images.
5. Sum up pixel values among x and y directory and find peaks and mean values.
6. Reject images with low mean pixel values or low peaks number.
7. Skip images with extremely high mean pixels values.
8. Crop images based on peaks.
