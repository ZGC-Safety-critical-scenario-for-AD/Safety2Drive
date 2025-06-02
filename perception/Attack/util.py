import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import io
from contextlib import redirect_stdout
from typing import Optional, List, Tuple
import torch

from vehicle_detection.utils.metrics import ConfusionMatrix


def build_projection_matrix(w, h, fov):
    focal = w / (2. * np.tan(fov * np.pi / 360.))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.
    K[1, 2] = h / 2.

    return K

def get_image_point(loc, K, w2c):
    point = np.array([loc.x, loc.y, loc.z, 1])

    point_camera = np.dot(w2c, point)

    point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

    point_img = np.dot(K, point_camera)

    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]

    return point_img[0:2]

def eval_coco_metrics(anno_json: str, pred_json: str, txt_save_path: str, w_mode: str = "a") -> np.ndarray:
    """Compare and eval pred json producing coco metrics."""
    anno = COCO(anno_json)  # init annotations api
    pred = anno.loadRes(pred_json)  # init predictions api
    evaluator = COCOeval(anno, pred, "bbox")

    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    # capture evaluator stats and save to file
    std_out = io.StringIO()
    with redirect_stdout(std_out):
        evaluator.summarize()
    eval_stats = std_out.getvalue()
    with open(txt_save_path, w_mode, encoding="utf-8") as fwriter:
        fwriter.write(eval_stats)
    return evaluator.stats

def calc_asr(
    boxes,
    boxes_pred,
    class_list: List[str],
    lo_area: float = 20**2,
    hi_area: float = 67**2,
    cls_id: Optional[int] = None,
    class_agnostic: bool = False,
    recompute_asr_all: bool = False,
) -> Tuple[float, float, float, float]:
    """
    Calculate attack success rate (How many bounding boxes were hidden from the detector) for all predictions and
    for different bbox areas.

    Note cls_id is None, misclassifications are ignored and only missing detections are considered attack success.
    Args:
        boxes: torch.Tensor, first pass boxes (gt unpatched boxes) [class, x1, y1, x2, y2]
        boxes_pred: torch.Tensor, second pass boxes (patched boxes) [x1, y1, x2, y2, conf, class]
        class_list: list of class names in correct order
        lo_area: small bbox area threshold
        hi_area: large bbox area threshold
        cls_id: filter for a particular class
        class_agnostic: All classes are considered the same
        recompute_asr_all: Recomputer ASR for all boxes aggregated together slower but more acc. asr
    Return:
        attack success rates bbox area tuple: small, medium, large, all
            float, float, float, float
    """
    # if cls_id is provided and evaluation is not class agnostic then mis-clsfs count as attack success
    if cls_id is not None:
        boxes = boxes[boxes[:, 0] == cls_id]
        boxes_pred = boxes_pred[boxes_pred[:, 5] == cls_id]

    boxes_area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 4] - boxes[:, 2])
    boxes_pred_area = (boxes_pred[:, 2] - boxes_pred[:, 0]) * (boxes_pred[:, 3] - boxes_pred[:, 1])

    b_small = boxes[boxes_area < lo_area]
    bp_small = boxes_pred[boxes_pred_area < lo_area]
    b_med = boxes[torch.logical_and(boxes_area <= hi_area, boxes_area >= lo_area)]
    bp_med = boxes_pred[torch.logical_and(boxes_pred_area <= hi_area, boxes_pred_area >= lo_area)]
    b_large = boxes[boxes_area > hi_area]
    bp_large = boxes_pred[boxes_pred_area > hi_area]
    assert (bp_small.shape[0] + bp_med.shape[0] + bp_large.shape[0]) == boxes_pred.shape[0]
    assert (b_small.shape[0] + b_med.shape[0] + b_large.shape[0]) == boxes.shape[0]

    conf_matrix = ConfusionMatrix(len(class_list))
    conf_matrix.process_batch(bp_small, b_small)
    tps_small, fps_small = conf_matrix.tp_fp()
    conf_matrix = ConfusionMatrix(len(class_list))
    conf_matrix.process_batch(bp_med, b_med)
    tps_med, fps_med = conf_matrix.tp_fp()
    conf_matrix = ConfusionMatrix(len(class_list))
    conf_matrix.process_batch(bp_large, b_large)
    tps_large, fps_large = conf_matrix.tp_fp()
    if recompute_asr_all:
        conf_matrix = ConfusionMatrix(len(class_list))
        conf_matrix.process_batch(boxes_pred, boxes)
        tps_all, fps_all = conf_matrix.tp_fp()
    else:
        tps_all, fps_all = tps_small + tps_med + tps_large, fps_small + fps_med + fps_large

    # class agnostic mode (Mis-clsfs are ignored, only non-dets matter)
    if class_agnostic:
        tp_small = tps_small.sum() + fps_small.sum()
        tp_med = tps_med.sum() + fps_med.sum()
        tp_large = tps_large.sum() + fps_large.sum()
        tp_all = tps_all.sum() + fps_all.sum()
    # filtering by cls_id or non class_agnostic mode (Mis-clsfs are successes)
    elif cls_id is not None:  # consider single class, mis-clsfs or non-dets
        tp_small = tps_small[cls_id]
        tp_med = tps_med[cls_id]
        tp_large = tps_large[cls_id]
        tp_all = tps_all[cls_id]
    else:  # non class_agnostic, mis-clsfs or non-dets
        tp_small = tps_small.sum()
        tp_med = tps_med.sum()
        tp_large = tps_large.sum()
        tp_all = tps_all.sum()

    asr_small = 1.0 - tp_small / (b_small.shape[0] + 1e-6)
    asr_medium = 1.0 - tp_med / (b_med.shape[0] + 1e-6)
    asr_large = 1.0 - tp_large / (b_large.shape[0] + 1e-6)
    asr_all = 1.0 - tp_all / (boxes.shape[0] + 1e-6)

    return max(asr_small, 0.0), max(asr_medium, 0.0), max(asr_large, 0.0), max(asr_all, 0.0)

def in_gt_bbox(x, y, w, h, gt_bbox_coordinates):
    '''
    Checks whether a box with top left coordinate (x,y) and bottom right coordinate (x+w, y+h) will fall in a GT bbox region

    Inputs:
    - x, y: coordinate of the point
    - w, h: width and height of the trigger
    - GT_bbox_coordinates: list of shape (N,4), where N represent the number of objects in the image, and the 4 elements in the second 
                           dimension are (class_name, x_min, y_min, x_max, y_max). The GT bbox coordinates are needed to ensure that the triggers will 
                           not be scattered into the GT bbox area.

    Return:
    Ture or False
    '''
    for eachbbox in gt_bbox_coordinates:
        x_min, y_min, x_max, y_max = eachbbox[0], eachbbox[1], eachbbox[2], eachbbox[3]
        width_in = x <= x_max and x >= x_min or x+w <= x_max and x+w >= x_min
        height_in = y <= y_max and y >= y_min or y+h <= y_max and y+h >= y_min
        if width_in or height_in:
            return True
    return False

def scatter_trigger(image, trigger, fusion_ratio, gt_bbox_coordinates, num_trigger, seed=3407):
    '''
    The function for scattering triggers randomly on an input image. Used for the current OGA testing. 
    See Inputs and Outputs for usage detail.

    Inputs:
    - image: numpy array of shape (H,W,C), where W and H is the width and height of the image, C is the number of channels.
    - trigger: numpy array of shape (Ht, Wt, C) 
        - *WARNING*: the function only checks whether the trigger size is smaller than the image size. But for practical usage, I recommand that
                the trigger size should be way smaller that the image size. This can not only help make it less visually obvious, but also if
                the trigger size is too large, it may be impossible to insert the trigger and the function can get stuck since we don't allow 
                the trigger to overlap with GT bboxes.
    - fusion_ratio: float scalar; the value used when fusing the original pixel value with the trigger content. 
                    More concretely, for each channel c, poisoned_image[?,?,c] = (1-fusion_ratio)*image[?,?,c] + fusion_ratio*trigger[?,?,c]
                    The larger the fusion_ratio, the more visible the trigger is.
    - GT_bbox_coordinates: list of shape (N,4), where N represent the number of objects in the image, and the 4 elements in the second 
                           dimension are (x_min, y_min, x_max, y_max). The GT bbox coordinates are needed to ensure that the triggers will 
                           not be scattered into the GT bbox area.
    - num_trigger: the number of triggers to be scattered into the image.
    - seed: the random seed. Used for reproducibility. default=3407. Pass None if no reproducibility is needed.

    Outputs:
    - poisoned_image: The poisoned image with 'num_trigger' triggers randomly scattered in the background. The shape is the same 
                      as the input image.
    - num_trigger_added: number of triggers we added 
    '''
    H, W, C = image.shape
    Ht, Wt, C = trigger.shape

    assert image.shape[2] == trigger.shape[2], "The number of channels of the image is not the same as the trigger"
    assert W > Wt and H > Ht, "trigger size is bigger than input image size"
    assert fusion_ratio <= 1 and fusion_ratio >= 0, "The fusion ratio must be between 0 and 1"

    if seed != None:
        np.random.seed(seed)

    poisoned_image = image.copy()

    num_trigger_added = 0
    triggers_put = []
    iter = 0
    while True:
        iter += 1
        if num_trigger_added == num_trigger:
            break
        elif iter >= 500:
            break

        x_pos = np.random.randint(0, W-Wt+1)
        y_pos = np.random.randint(0, H-Ht+1)
        if not in_gt_bbox(x_pos, y_pos, Wt, Ht, gt_bbox_coordinates):
            poisoned_image[y_pos:y_pos+Ht, x_pos:x_pos+Wt, :] = (
                1-fusion_ratio)*poisoned_image[y_pos:y_pos+Ht, x_pos:x_pos+Wt, :] + fusion_ratio*trigger
            # We don't want triggers to overlap with each other
            gt_bbox_coordinates.append([x_pos, y_pos, x_pos+Wt, y_pos+Ht])
            triggers_put.append([x_pos, y_pos, Wt, Ht])
            num_trigger_added += 1

    return poisoned_image, triggers_put

def yolo_2_xml_bbox_coco(names, bbox, w, h):
    w_half_len = (float(bbox[3]) * w) / 2
    h_half_len = (float(bbox[4]) * h) / 2

    xmin = int((float(bbox[1]) * w) - w_half_len)
    ymin = int((float(bbox[2]) * h) - h_half_len)
    xmax = int((float(bbox[1]) * w) + w_half_len)
    ymax = int((float(bbox[2]) * h) + h_half_len)
    
    return [names[int(bbox[0])], xmin, ymin, xmax, ymax]

def point_range_filter(pts, point_range=[0, -39.68, -3, 69.12, 39.68, 1]):
    '''
    data_dict: dict(pts, gt_bboxes_3d, gt_labels, gt_names, difficulty)
    point_range: [x1, y1, z1, x2, y2, z2]
    '''
    flag_x_low = pts[:, 0] > point_range[0]
    flag_y_low = pts[:, 1] > point_range[1]
    flag_z_low = pts[:, 2] > point_range[2]
    flag_x_high = pts[:, 0] < point_range[3]
    flag_y_high = pts[:, 1] < point_range[4]
    flag_z_high = pts[:, 2] < point_range[5]
    keep_mask = flag_x_low & flag_y_low & flag_z_low & flag_x_high & flag_y_high & flag_z_high
    pts = pts[keep_mask]
    return pts 
