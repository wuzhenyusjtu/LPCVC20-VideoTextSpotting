# Adaptive sampling
Directory includes codes for solution for LPCV 2020 competition, which includes non-text frame temporal rejection and 
text region aware spatial cropping, as well as adaptive sampling method OCR detection based on EAST model and recognition 
based on CRNN model.
## Installation
pip install -r requirements.txt
## Usage
python main.py input_video_path query_file_path
### Implementation
1. Initialize sample rate
2. Input image pass through selection, which is the same as non-text frame temporal rejection and text region aware 
spatial cropping.
3. If image rejected by selector, set sample rate to 'NT'.
4. Pass image through detection. If image rejected by detector, set sample rate to 'LQ'.
5. Pass image through recognition. If image rejected by recognizer, set sample rate to 'LQ'.
6. If current sample rate is 'HQ', and recognized text result equals cache results, clear cache results and set sample 
rate to 'NT'.
7.  If current sample rate is not 'HQ', and there are text recognized, cache the result to cache result, set sample rate 
to 'HQ'.
8. Repeat 2 to 7.