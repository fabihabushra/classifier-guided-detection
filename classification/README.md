# AG-CNN
This repository contains model inference code for the classification framework accompanying the paper "Deep Learning in Computed Tomography Pulmonary Angiography Imaging: A Dual-Pronged Approach for Pulmonary Embolism Detection". 

<img src="https://github.com/fabihabushra/classifier-guided-detection/assets/48798437/393aca12-e423-40e6-808c-c0ede2950c8b" alt="architecture_AGCNN-01" style="width: 85%;">

## Environment
Please prepare an environment with python=3.9, and then use the following command for the dependencies:

```bash
pip install -r requirements.txt
```

## Directory Structure
```bash
.
├── classification
│   ├── data
│   │   └── Test
│   │       └── *.png
│   ├── labels
│   │   └── test_list.txt
│   ├── weights
│   │   ├── densenet_121
│   │   ├── densenet_201
│   │   ├── inception_v3
│   │   ├── mobilenet_v3_large
│   │   ├── resnet_152
│   │   └── resnet_50
│   ├── test.py
│   └── ...
```

## Test Data
Use the [preprocessed test data](https://drive.google.com/drive/folders/1s8_42BF2beqi9FjlMyK5UBAiKQiojOhC?usp=sharing) for model inference.

## Model Weights
You can download the trained model weights from [this link](https://drive.google.com/drive/folders/1HwDC2aD_1qlhRx2mOMkzpXLGvepuxjvf?usp=sharing).

## Model Inference
To perform model inference on the test data, use the following command:
```bash
python test.py --model inception_v3
```
Other available models for inference:
- densenet_121
- densenet_201
- mobilenet_v3_large
- resnet_152
- resnet_50



