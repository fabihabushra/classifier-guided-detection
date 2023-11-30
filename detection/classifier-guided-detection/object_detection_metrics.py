import os
import torch
import numpy as np
import argparse
from torchmetrics.detection import mean_ap
from torchvision.datasets.voc import VOCDetection
from pathlib import Path


def load_yolo_annotations(file_path):
    annotations = []

    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            annotation = {}
            line = line.strip().split(' ')
            class_label = line[0]
            x_centre, y_centre, width, height = map(float, line[1:5])

            x_min = x_centre - width / 2
            y_min = y_centre - height / 2
            x_max = x_centre + width / 2
            y_max = y_centre + height / 2

            try:
                scores = float(line[5])
                annotation = {
                    'class_label': class_label,
                    'bbox': [x_min, y_min, x_max, y_max],
                    'scores': scores
                }
            except:
                annotation = {
                    'class_label': class_label,
                    'bbox': [x_min, y_min, x_max, y_max],
                }

            annotations.append(annotation)

    return annotations

def convert_to_detection_tensor(annotations, include_scores=False):
    detection_tensors = []
    class_mapping = {}
    ann = annotations
    bbox = ann['bbox']
    class_label = ann.get('class_label', '')
    class_label = [int(x) for x in class_label]


    detection_tensor = {
        'boxes': torch.tensor(bbox, dtype=torch.float32),
        'labels': torch.tensor(class_label, dtype=torch.int64)
    }
    if include_scores:
        scores = ann.get('scores', 0.0)
        detection_tensor['scores'] = torch.tensor(scores, dtype=torch.float32)
    detection_tensors.append(detection_tensor)
    return detection_tensors


def calculate_metrics(ground_truth_dir, predicted_dir, ground_truth_image_dir):
    ground_truth = {}
    predicted = {}

    # Load ground truth annotations
    for image_file in os.listdir(ground_truth_image_dir):
        image_name = image_file.split('.')[0]
        file_path = os.path.join(ground_truth_dir, image_name + '.txt')
        
        if not os.path.exists(file_path):
            # Simulate the absence of a ground truth label file
            ground_truth[image_name] = {
                'class_label': [],
                'bbox': []
            }
        else:
            annotations = load_yolo_annotations(file_path)
            merged_ann = {
                'class_label': [],
                'bbox': []
            }
            for ann in annotations:
                merged_ann['class_label'].append(ann['class_label'])
                merged_ann['bbox'].append(ann['bbox'])

            ground_truth[image_name] = merged_ann

        # Load predicted annotations
        predicted_file_path = os.path.join(predicted_dir, image_name + '.txt')
        
        if not os.path.exists(predicted_file_path):
            # Simulate the absence of a predicted label file
            predicted[image_name] = {
                'class_label': [],
                'bbox': [],
                'scores': []
            }
        else:
            annotations_pred = load_yolo_annotations(predicted_file_path)
            merged_ann_pred = {
                'class_label': [],
                'bbox': [],
                'scores': []
            }
            for ann_pred in annotations_pred:
                merged_ann_pred['class_label'].append(ann_pred['class_label'])
                merged_ann_pred['bbox'].append(ann_pred['bbox'])
                merged_ann_pred['scores'].append(ann_pred['scores'])

            predicted[image_name] = merged_ann_pred

    # Convert annotations to tensors
    ground_truth_tensors = []
    predicted_tensors = []

    for gt_ann in ground_truth.values():
        ground_truth_tensors.extend(convert_to_detection_tensor(gt_ann, include_scores=False))

    for pred_ann in predicted.values():
        predicted_tensors.extend(convert_to_detection_tensor(pred_ann, include_scores=True))

    metric_1 = mean_ap.MeanAveragePrecision(iou_thresholds=[0.1])
    metric_2 = mean_ap.MeanAveragePrecision()
    metric_1.update(predicted_tensors, ground_truth_tensors)
    metric_2.update(predicted_tensors, ground_truth_tensors)
    mAP_1 = metric_1.compute()
    mAP_2 = metric_2.compute()

    return mAP_1, mAP_2

def load_coco_annotations(file_path):
    annotations = []

    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            annotation = {}
            line = line.strip().split(' ')
            class_label = line[0]
            x_min, y_min, x_max, y_max = map(float, line[1:5])

            # x_min = x_centre - width / 2
            # y_min = y_centre - height / 2
            # x_max = x_centre + width / 2
            # y_max = y_centre + height / 2

            try:
                scores = float(line[5])
                annotation = {
                    'class_label': class_label,
                    'bbox': [x_min, y_min, x_max, y_max],
                    'scores': scores
                }
            except:
                annotation = {
                    'class_label': class_label,
                    'bbox': [x_min, y_min, x_max, y_max],
                }

            annotations.append(annotation)

    return annotations


class ConfusionMatrix:
    """
    A class for calculating and updating a confusion matrix for object detection and classification tasks.

    Attributes:
        task (str): The type of task, either 'detect' or 'classify'.
        matrix (np.array): The confusion matrix, with dimensions depending on the task.
        nc (int): The number of classes.
        conf (float): The confidence threshold for detections.
        iou_thres (float): The Intersection over Union threshold.
    """

    def __init__(self, nc, iou_thres=0.1, task='detect'):
        """Initialize attributes for the YOLO model."""
        self.task = task
        self.matrix = np.zeros((nc + 1, nc + 1)) if self.task == 'detect' else np.zeros((nc, nc))
        self.nc = nc  # number of classes
        self.iou_thres = iou_thres

    def process_cls_preds(self, preds, targets):
        """
        Update confusion matrix for classification task

        Args:
            preds (Array[N, min(nc,5)]): Predicted class labels.
            targets (Array[N, 1]): Ground truth class labels.
        """
        preds, targets = torch.cat(preds)[:, 0], torch.cat(targets)
        for p, t in zip(preds.cpu().numpy(), targets.cpu().numpy()):
            self.matrix[p][t] += 1

    def process_batch(self, detections, labels, conf):
        """
        Update confusion matrix for object detection task.

        Args:
            detections (Array[N, 6]): Detected bounding boxes and their associated information.
                                      Each row should contain (class, x1, y1, x2, y2, conf).
            labels (Array[M, 5]): Ground truth bounding boxes and their associated class labels.
                                  Each row should contain (class, x1, y1, x2, y2).
        """
        if detections is None:
            gt_classes = labels.int()
            for gc in gt_classes:
                self.matrix[self.nc, gc] += 1  # background FN
            return

        detections = detections[detections[:, 5] > conf]
        gt_classes = labels[:, :]
        gt_classes = torch.from_numpy(gt_classes).int()
        detection_classes = detections[:, 0]
        detection_classes = torch.from_numpy(detection_classes).int()
        # Convert bounding box coordinates from x_centre, y_centre, width, height to xmin, ymin, xmax, ymax format
        labels_coordinates = labels[:, 1:]
        labels_coordinates[:, 0] = labels[:, 1] - labels[:, 3] / 2  # xmin = x_centre - width/2
        labels_coordinates[:, 1] = labels[:, 2] - labels[:, 4] / 2  # ymin = y_centre - height/2
        labels_coordinates[:, 2] = labels[:, 1] + labels[:, 3] / 2  # xmax = x_centre + width/2
        labels_coordinates[:, 3] = labels[:, 2] + labels[:, 4] / 2  # ymax = y_centre + height/2

        detections_coordinates = detections[:, 1:5]
        detections_coordinates[:, 0] = detections[:, 1] - detections[:, 3] / 2  # xmin = x_centre - width/2
        detections_coordinates[:, 1] = detections[:, 2] - detections[:, 4] / 2  # ymin = y_centre - height/2
        detections_coordinates[:, 2] = detections[:, 1] + detections[:, 3] / 2  # xmax = x_centre + width/2
        detections_coordinates[:, 3] = detections[:, 2] + detections[:, 4] / 2  # ymax = y_centre + height/2

        iou = box_iou(labels_coordinates, detections_coordinates)

        x = torch.where(iou > self.iou_thres)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(int)
        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and sum(j) == 1:
                self.matrix[detection_classes[m1[j]], gc] += 1  # correct
            else:
                self.matrix[self.nc, gc] += 1  # true background

        if n:
            for i, dc in enumerate(detection_classes):
                if not any(m1 == i):
                    self.matrix[dc, self.nc] += 1  # predicted background

    def matrix(self):
        """Returns the confusion matrix."""
        return self.matrix

    def tp_fp(self):
        """Returns true positives and false positives."""
        tp = self.matrix.diagonal()  # true positives
        fp = self.matrix.sum(1) - tp  # false positives
        fn = self.matrix.sum(0) - tp  # false negatives (missed detections)
        return (tp[:-1], fp[:-1], fn[:-1]) if self.task == 'detect' else (tp, fp, fn)  # remove background class if task=detect

    def get_confidence_scores(self, detections):
        """
        Extract confidence scores from detections.

        Args:
            detections (Array[N, 6]): Detected bounding boxes and their associated information.
                                      Each row should contain (class, x1, y1, x2, y2, conf).

        Returns:
            confidence_scores (Array[N]): Array of confidence scores.
        """
        conf_score = detections[:, 5]
        return conf_score


def ap_per_class(tp,
                 conf,
                 pred_cls,
                 target_cls,
                 plot=False,
                 on_plot=None,
                 save_dir=Path(),
                 names=(),
                 eps=1e-16,
                 prefix=''):
    """
    Computes the average precision per class for object detection evaluation.

    Args:
        tp (np.ndarray): Binary array indicating whether the detection is correct (True) or not (False).
        conf (np.ndarray): Array of confidence scores of the detections.
        pred_cls (np.ndarray): Array of predicted classes of the detections.
        target_cls (np.ndarray): Array of true classes of the detections.
        plot (bool, optional): Whether to plot PR curves or not. Defaults to False.
        on_plot (func, optional): A callback to pass plots path and data when they are rendered. Defaults to None.
        save_dir (Path, optional): Directory to save the PR curves. Defaults to an empty path.
        names (tuple, optional): Tuple of class names to plot PR curves. Defaults to an empty tuple.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-16.
        prefix (str, optional): A prefix string for saving the plot files. Defaults to an empty string.

    Returns:
        (tuple): A tuple of six arrays and one array of unique classes, where:
            tp (np.ndarray): True positive counts for each class.
            fp (np.ndarray): False positive counts for each class.
            p (np.ndarray): Precision values at each confidence threshold.
            r (np.ndarray): Recall values at each confidence threshold.
            f1 (np.ndarray): F1-score values at each confidence threshold.
            ap (np.ndarray): Average precision for each class at different IoU thresholds.
            unique_classes (np.ndarray): An array of unique classes that have data.

    """

    # Sort by objectness
    i = np.argsort(-conf)
    print(f'i: {i}')
    print(f'tp[i]: {tp[i]}')
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # number of labels
        n_p = i.sum()  # number of predictions
        if n_p == 0 or n_l == 0:
            continue

        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # Recall
        recall = tpc / (n_l + eps)  # recall curve
        r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

        # Precision
        precision = tpc / (tpc + fpc)  # precision curve
        p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
            if plot and j == 0:
                py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + eps)
    names = [v for k, v in names.items() if k in unique_classes]  # list: only classes that have data
    names = dict(enumerate(names))  # to dict


    i = smooth(f1.mean(0), 0.1).argmax()  # max F1 index
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    return tp, fp, p, r, f1, ap, unique_classes.astype(int)


def box_iou(box1, box2, eps=1e-7):
    """
    Calculate intersection-over-union (IoU) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Based on https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py

    Args:
        box1 (torch.Tensor): A tensor of shape (N, 4) representing N bounding boxes.
        box2 (torch.Tensor): A tensor of shape (M, 4) representing M bounding boxes.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): An NxM tensor containing the pairwise IoU values for every element in box1 and box2.
    """
    # Convert box1 and box2 to PyTorch tensors
    box1 = torch.from_numpy(box1)
    box2 = torch.from_numpy(box2)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp_(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)

def compute_ap(recall, precision):
    """
    Compute the average precision (AP) given the recall and precision curves.

    Arguments:
        recall (list): The recall curve.
        precision (list): The precision curve.

    Returns:
        (float): Average precision.
        (np.ndarray): Precision envelope curve.
        (np.ndarray): Modified recall curve with sentinel values added at the beginning and end.
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x-axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec

def smooth(y, f=0.05):
    """Box filter of fraction f."""
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = np.ones(nf // 2)  # ones padding
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return np.convolve(yp, np.ones(nf) / nf, mode='valid')  # y-smoothed

def load_annotations(file_path):
    # Load annotations from a text file
    with open(file_path, 'r') as file:
        lines = file.read().splitlines()
    annotations = []
    for line in lines:
        data = line.split()
        class_label = int(data[0])
        x_center = float(data[1])
        y_center = float(data[2])
        width = float(data[3])
        height = float(data[4])
        try:
            conf = float(data[5])
            annotations.append([class_label, x_center, y_center, width, height, conf])
        except:
            annotations.append([class_label, x_center, y_center, width, height])
    return np.array(annotations)

def evaluate_object_detection(dir1, dir2):
    # Load ground truth annotations
    gt_annotations = {}
    for file_path in sorted(dir1.iterdir()):
        file_name = file_path.stem
        annotations = load_annotations(file_path)
        gt_annotations[file_name] = annotations

    # Load predicted annotations
    pred_annotations = {}
    for file_path in sorted(dir2.iterdir()):
        file_name = file_path.stem
        annotations = load_annotations(file_path)
        pred_annotations[file_name] = annotations


    # Initialize confusion matrix
    num_classes = len(set([c for _, annotations in gt_annotations.items() for c, *_ in annotations]))
    confusion_matrix = ConfusionMatrix(num_classes)

    # Process annotations
    for file_name, gt_annos in gt_annotations.items():
        if file_name in pred_annotations:
            pred_annos = pred_annotations[file_name]
            confusion_matrix.process_batch(pred_annos, gt_annos, conf=0.05)

    tp, fp, fn = confusion_matrix.tp_fp()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate metrics for object detection')
    parser.add_argument('--gnd_label_dir', type=Path, help='Path to ground truth label directory containing .txt files')
    parser.add_argument('--pred_label_dir', type=Path, help='Path to prediction label directory containing .txt files')
    parser.add_argument('--gnd_image_dir', type=Path, help='Path to ground truth image directory containing .png files')
    args = parser.parse_args()

    # Validate if directories exist
    if not args.gnd_label_dir.exists() or not args.pred_label_dir.exists() or not args.gnd_image_dir.exists():
        print("One or more directories do not exist. Please provide valid directories.")
    else:
    # Calculate mAP
        mAP_20, mAP_50 = calculate_metrics(args.gnd_label_dir, args.pred_label_dir, args.gnd_image_dir)

        print("map_20:", mAP_20['map'])
        print("map_50:", mAP_50['map_50'])

        # Evaluate other metrics
        precision, recall, f1_score = evaluate_object_detection(args.gnd_label_dir, args.pred_label_dir)

        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1_Score: {f1_score}')
