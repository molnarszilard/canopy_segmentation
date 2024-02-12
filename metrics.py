import numpy as np
import torch
import torch.nn as nn

class LossIoU(nn.Module):
    def __init__(self):
        super(LossIoU, self).__init__()

    def forward(self, pred, gt):
        # print("GT min,mean,max:%f, %f, %f"%(gt.min(),gt.mean(),gt.max()))        
        intersection_tensor=pred*gt
        intersection = torch.sum(intersection_tensor, dim = (0,1,2,3))
        union_tensor = pred+gt-intersection_tensor
        union = torch.sum(union_tensor, dim = (0,1,2,3))
        iou = intersection/union
        return 1-iou
    
class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, pred, gt):
        return torch.sqrt(torch.mean((pred-gt)**2))
    
class MAPMetric(nn.Module):
    def __init__(self):
        super(MAPMetric, self).__init__()

    def forward(self,predicted_masks,ground_truth_masks):
        iou_thresholds = np.linspace(0.5, 0.95, 10)  # For mAP50-95
        map50_95 = self.mean_average_precision(predicted_masks, ground_truth_masks, iou_thresholds)
        iou_thresholds = [0.5]  # For mAP50
        map50 = self.mean_average_precision(predicted_masks, ground_truth_masks, iou_thresholds)
        return map50,map50_95

def mask_to_bbox(mask):
    # Find the rows and columns where the mask has values
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    # Find the indices of the first and last row and column
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]

    # Return the coordinates of the bounding box
    return xmin, ymin, xmax, ymax


def calculate_precision_recall(predicted_bboxes, ground_truth_bboxes, iou_threshold):
    """
    Calculate precision and recall at a given IoU threshold.

    Parameters:
    predicted_bboxes: list of predicted bounding boxes
    ground_truth_bboxes: list of ground truth bounding boxes
    iou_threshold: float, IoU threshold to consider a prediction as true positive

    Returns:
    tuple: (precision, recall)
    """
    true_positives = 0
    false_positives = 0
    false_negatives = len(ground_truth_bboxes)

    for pred_bbox in predicted_bboxes:
        for gt_bbox in ground_truth_bboxes:
            iou = calculate_iou(pred_bbox, gt_bbox)
            if iou >= iou_threshold:
                true_positives += 1
                false_negatives -= 1
                break
        else:
            false_positives += 1

    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0

    return precision, recall


def calculate_iou(bbox1, bbox2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters:
    bbox1, bbox2: (xmin, ymin, xmax, ymax)

    Returns:
    float: IoU
    """
    # Determine the coordinates of the intersection rectangle
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])

    # Calculate area of intersection rectangle
    intersection_area = max(0, x_right - x_left + 1) * max(0, y_bottom - y_top + 1)

    # Calculate area of both bounding boxes
    bbox1_area = (bbox1[2] - bbox1[0] + 1) * (bbox1[3] - bbox1[1] + 1)
    bbox2_area = (bbox2[2] - bbox2[0] + 1) * (bbox2[3] - bbox2[1] + 1)

    # Calculate union area
    union_area = bbox1_area + bbox2_area - intersection_area

    # Compute IoU
    iou = intersection_area / float(union_area)
    return iou


def mean_average_precision(predicted_masks, ground_truth_masks, iou_thresholds):
    predicted_bboxes = [mask_to_bbox(mask) for mask in predicted_masks]
    ground_truth_bboxes = [mask_to_bbox(mask) for mask in ground_truth_masks]

    average_precisions = []
    for iou_threshold in iou_thresholds:
        precision, recall = calculate_precision_recall(predicted_bboxes, ground_truth_bboxes, iou_threshold)
        average_precisions.append(np.mean(precision))  # Assuming precision is a list

    return np.mean(average_precisions)