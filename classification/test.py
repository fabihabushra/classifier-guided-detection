# encoding: utf-8

"""
The main AG-CNN model implementation.
"""
import re
import sys
import os
import cv2
import time
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import csv
import pandas as pd
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
import argparse
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from utils import EmbolismDataSet
from sklearn.metrics import roc_auc_score, roc_curve, auc
from skimage.measure import label
from model import Densenet121_AG, Densenet201_AG, ResNet50_AG, ResNet152_AG, InceptionV3_AG, mobilenet_v3_large, Fusion_Branch
from PIL import Image
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch
from mpl_toolkits.axes_grid1.inset_locator import mark_inset



parser = argparse.ArgumentParser(description='Load model checkpoint paths')
parser.add_argument('--model', type=str, default='inception_v3', choices=['densenet_121', 'densenet_201', 'resnet_50', 'resnet_152', 'inception_v3', 'mobilenet_v3_large'], help='Model to load')
args = parser.parse_args()

model_to_load = args.model

# load the weights of model for inference
CKPT_PATH_G = f"./weights/{model_to_load}/Global_epoch.pkl" 
CKPT_PATH_L = f"./weights/{model_to_load}/Local_epoch.pkl" 
CKPT_PATH_F = f"./weights/{model_to_load}/Fusion_epoch.pkl"

# load test dataset and label list path
TEST_DIR = './data/Test'
TEST_IMAGE_LIST = './labels/test_list.txt'

test_data_mean = 0.1966
test_data_std = 0.2856
BATCH_SIZE = 16
N_CLASSES = 2
CLASS_NAMES = [ 'Pulmonary Embolism', 'Non-Pulmonary Embolism']

normalize_test = transforms.Normalize(
  mean=[test_data_mean, test_data_mean, test_data_mean],
  std=[test_data_std, test_data_std, test_data_std]
)

preprocess = transforms.Compose([
   transforms.Resize((256,256)),
   transforms.CenterCrop(224),
   transforms.ToTensor(),
   normalize_test,
])


def Attention_gen_patchs(ori_image, fm_cuda):
    # fm => mask =>(+ ori-img) => crop = patchs
    feature_conv = fm_cuda.data.cpu().numpy()
    size_upsample = (224, 224) 
    bz, nc, h, w = feature_conv.shape

    patchs_cuda = torch.FloatTensor().cuda()

    for i in range(0, bz):
        feature = feature_conv[i]
        cam = feature.reshape((nc, h*w))
        cam = cam.sum(axis=0)
        cam = cam.reshape(h,w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)

        heatmap_bin = binImage(cv2.resize(cam_img, size_upsample))
        heatmap_maxconn = selectMaxConnect(heatmap_bin)
        heatmap_mask = heatmap_bin * heatmap_maxconn

        ind = np.argwhere(heatmap_mask != 0)
        minh = min(ind[:,0])
        minw = min(ind[:,1])
        maxh = max(ind[:,0])
        maxw = max(ind[:,1])
        
        # to ori image 
        image = ori_image[i].numpy().reshape(224,224,3)
        image = image[int(224*0.334):int(224*0.667),int(224*0.334):int(224*0.667),:]

        image = cv2.resize(image, size_upsample)
        image_crop = image[minh:maxh,minw:maxw,:] * 256     # because image was normalized before
        image_crop = preprocess(Image.fromarray(image_crop.astype('uint8')).convert('RGB')) 

        img_variable = torch.autograd.Variable(image_crop.reshape(3,224,224).unsqueeze(0).cuda())

        patchs_cuda = torch.cat((patchs_cuda,img_variable),0)

    return patchs_cuda


def binImage(heatmap):
    _, heatmap_bin = cv2.threshold(heatmap , 0 , 255 , cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    return heatmap_bin


def selectMaxConnect(heatmap):
    labeled_img, num = label(heatmap, connectivity=2, background=0, return_num=True)    
    max_label = 0
    max_num = 0
    for i in range(1, num+1):
        if np.sum(labeled_img == i) > max_num:
            max_num = np.sum(labeled_img == i)
            max_label = i
    lcc = (labeled_img == max_label)
    if max_num == 0:
       lcc = (labeled_img == -1)
    lcc = lcc + 0
    return lcc 


def main():
    print('********************loading data********************')
    test_dataset = EmbolismDataSet(data_dir=TEST_DIR,
                                    image_list_file=TEST_IMAGE_LIST,
                                    transform=transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        normalize_test,
                                    ]))
    test_loader = DataLoader(dataset=test_dataset, batch_size=8,
                             shuffle=False, num_workers=1, pin_memory=True)


    print('********************loading model********************')
    if model_to_load == 'densenet_201':
        Global_Branch_model = Densenet201_AG(pretrained = True, num_classes = N_CLASSES).cuda()
        Local_Branch_model = Densenet201_AG(pretrained = True, num_classes = N_CLASSES).cuda()
        Fusion_Branch_model = Fusion_Branch(input_size = 3840, output_size = N_CLASSES).cuda()

    elif model_to_load == 'densenet_121':
        Global_Branch_model = Densenet121_AG(pretrained = True, num_classes = N_CLASSES).cuda()
        Local_Branch_model = Densenet121_AG(pretrained = True, num_classes = N_CLASSES).cuda()
        Fusion_Branch_model = Fusion_Branch(input_size = 2048, output_size = N_CLASSES).cuda()

    elif model_to_load == 'resnet_50':
        Global_Branch_model = ResNet50_AG(pretrained = True, num_classes = N_CLASSES).cuda()
        Local_Branch_model = ResNet50_AG(pretrained = True, num_classes = N_CLASSES).cuda()
        Fusion_Branch_model = Fusion_Branch(input_size = 4096, output_size = N_CLASSES).cuda()

    elif model_to_load == 'resnet_152':
        Global_Branch_model = ResNet152_AG(pretrained = True, num_classes = N_CLASSES).cuda()
        Local_Branch_model = ResNet152_AG(pretrained = True, num_classes = N_CLASSES).cuda()
        Fusion_Branch_model = Fusion_Branch(input_size = 4096, output_size = N_CLASSES).cuda()

    elif model_to_load == 'inception_v3':
        Global_Branch_model = InceptionV3_AG(pretrained = True, num_classes = N_CLASSES).cuda()
        Local_Branch_model = InceptionV3_AG(pretrained = True, num_classes = N_CLASSES).cuda()
        Fusion_Branch_model = Fusion_Branch(input_size = 4096, output_size = N_CLASSES).cuda()

    elif model_to_load == 'mobilenet_v3_large':
        Global_Branch_model = mobilenet_v3_large(pretrained = True, num_classes = N_CLASSES).cuda()
        Local_Branch_model = mobilenet_v3_large(pretrained = True, num_classes = N_CLASSES).cuda()
        Fusion_Branch_model = Fusion_Branch(input_size = 1920, output_size = N_CLASSES).cuda()

    else:
        print('Model loading failed. Please check "model_to_load". ')

    if os.path.isfile(CKPT_PATH_G):
        checkpoint = torch.load(CKPT_PATH_G)
        Global_Branch_model.load_state_dict(checkpoint)
        print("=> loaded Global_Branch_model checkpoint")

    if os.path.isfile(CKPT_PATH_L):
        checkpoint = torch.load(CKPT_PATH_L)
        Local_Branch_model.load_state_dict(checkpoint)
        print("=> loaded Local_Branch_model checkpoint")

    if os.path.isfile(CKPT_PATH_F):
        checkpoint = torch.load(CKPT_PATH_F)
        Fusion_Branch_model.load_state_dict(checkpoint)
        print("=> loaded Fusion_Branch_model checkpoint")

    cudnn.benchmark = True
    print(f'\n################begin testing for AG-CNN with {model_to_load} backbone################')
    test(Global_Branch_model, Local_Branch_model, Fusion_Branch_model, test_loader)

def test(model_global, model_local, model_fusion, test_loader):

    # initialize the ground truth and output tensor
    gt = torch.FloatTensor().cuda()
    pred_global = torch.FloatTensor().cuda()
    pred_local = torch.FloatTensor().cuda()
    pred_fusion = torch.FloatTensor().cuda()

    # switch to evaluate mode
    model_global.eval()
    model_local.eval()
    model_fusion.eval()
    cudnn.benchmark = True
    pbar = tqdm(test_loader, ncols=100, leave=False)

    running_corrects1 = 0.0
    running_corrects2 = 0.0
    running_corrects3 = 0.0

    fusion_preds = []
    fusion_targets = []

    global_preds = []
    global_targets = []

    local_preds = []
    local_targets = []    

    all_image_names = []

    for i, (inp, target) in enumerate(pbar):
        with torch.no_grad():
            target = target.cuda()
            gt = torch.cat((gt, target), 0)
            input_var = torch.autograd.Variable(inp.cuda())

            output_global, fm_global, pool_global = model_global(input_var)
            
            patchs_var = Attention_gen_patchs(inp,fm_global)

            output_local, _, pool_local = model_local(patchs_var)

            output_fusion = model_fusion(pool_global,pool_local)

            fusion_preds += output_fusion.argmax(dim=1).tolist()
            fusion_targets += target.argmax(dim=1).tolist()

            global_preds += output_global.argmax(dim=1).tolist()
            global_targets += target.argmax(dim=1).tolist()

            local_preds += output_local.argmax(dim=1).tolist()
            local_targets += target.argmax(dim=1).tolist()           

            pred_global = torch.cat((pred_global, output_global.data), 0)
            pred_local = torch.cat((pred_local, output_local.data), 0)
            pred_fusion = torch.cat((pred_fusion, output_fusion.data), 0)

            _, preds1 = torch.max(output_fusion, 1)
            _, preds1_target = torch.max(target, 1)
            running_corrects1 += torch.sum(preds1 == preds1_target)

            _, preds2 = torch.max(output_global, 1)
            _, preds2_target = torch.max(target, 1)
            running_corrects2 += torch.sum(preds2 == preds2_target)

            _, preds3 = torch.max(output_local, 1)
            _, preds3_target = torch.max(target, 1)
            running_corrects3 += torch.sum(preds3 == preds3_target)

    test_fusion_acc = running_corrects1.double() / len(test_loader.dataset)
    test_global_acc = running_corrects2.double() / len(test_loader.dataset)
    test_local_acc = running_corrects3.double() / len(test_loader.dataset)

    AUROCs_g = compute_AUCs(gt, pred_global)
    AUROCs_g_PE = AUROCs_g[0]
    AUROCs_g_NPE = AUROCs_g[1]
    print('\nGLOBAL BRANCH:')
    for i in range(N_CLASSES):
        print(f'The AUROC of {CLASS_NAMES[i]} is {AUROCs_g[i]:.5f}')

    AUROCs_l = compute_AUCs(gt, pred_local)
    AUROCs_l_PE = AUROCs_l[0]
    AUROCs_l_NPE = AUROCs_l[1]
    print('\nLOCAL BRANCH:')
    for i in range(N_CLASSES):
        print(f'The AUROC of {CLASS_NAMES[i]} is {AUROCs_l[i]:.5f}')

    AUROCs_f = compute_AUCs(gt, pred_fusion)
    AUROCs_f_PE = AUROCs_f[0]
    AUROCs_f_NPE = AUROCs_f[1]
    print('\nFUSION BRANCH:')
    for i in range(N_CLASSES):
        print(f'The AUROC of {CLASS_NAMES[i]} is {AUROCs_f[i]:.5f}')


def compute_AUCs(gt, pred):
    """Computes Area Under the Curve (AUC) from prediction scores.

    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          can either be probability estimates of the positive class,
          confidence values, or binary decisions.

    Returns:
        List of AUROCs of all classes.
    """
    AUROCs = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    for i in range(N_CLASSES):
      try:
        AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
      except:
        print("No class present")
        AUROCs.append(0)
    return AUROCs


if __name__ == '__main__':
    main()