# Classifier-guided Detection
This directory contains the evaluation code for the classifier-guided detection accompanying the paper "Deep Learning in Computed Tomography Pulmonary Angiography Imaging: A Dual-Pronged Approach for Pulmonary Embolism Detection". 

## Environment
Please prepare an environment with python=3.9, and then use the following command for the dependencies:

```bash
pip install -r requirements.txt
```

## Directory Structure
```bash
.
├── detection
│   ├── data
│   │   └── images
│   │       └── *.png
│   │   └── labels
│   │       └── *.txt
│   ├── classifier_guided_prediction
│   │   └── classifier_guided_detection_nms
│   │       └── *.txt
│   │   └── classifier_guided_detection_nmw
│   │       └── *.txt
│   │   └── classifier_guided_detection_wbf
│   │       └── *.txt
│   ├── object_detection_metrics.py
```

## Test Data
Use the [preprocessed test data](https://drive.google.com/drive/folders/1D6Tnq2aShjFeA9l1kvWzYV30fTY0-51D?usp=drive_link) for detection performance evaluation.


## Usage
To compute object detection metrics using the predictions obtained from the classifier-guided detection, execute the following command:
```bash
python object_detection_metrics.py \
  --gnd_label_dir  "./data/labels/test" \
  --pred_label_dir "./classifier_guided_prediction/classifier_guided_detection_wbf" \
  --gnd_image_dir "./data/images/test" 
```
