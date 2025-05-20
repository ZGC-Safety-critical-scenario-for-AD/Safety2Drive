import cv2
import os
import torch
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

import sys
root_path = os.path.dirname(__file__)
sys.path.append(root_path)

from models.experimental import attempt_load
import numpy as np
from vehicle_detection.utils.general import non_max_suppression, scale_boxes, xyxy2xywh


class VehicleDetector():
    """Constructs a Vehicle Detector class using YOLOv5.

    Args:
        None
    """
    def __init__(self):
        model_name = 'yolov5s.pt' # yolov5_wtc yolov5s_cjc yolov8_oga yolov85_oga yolov5s_ljz best
        checkpoint_path = os.path.join(root_path, model_name)
        if 'v5' in model_name:
            self.model = attempt_load(checkpoint_path, device='cuda')
        elif 'v8' in model_name:
            self.model = YOLO(checkpoint_path)
        self.model_name = model_name

    def get_vehicles(self, ori_img):
        if 'v5' in self.model_name:
            img = cv2.resize(ori_img, (640, 640))
            img = img / 255.
            img = img[:, :, ::-1].transpose((2, 0, 1))
            img = np.expand_dims(img, axis=0)
            img = torch.from_numpy(img.copy()).float().to('cuda')
        
            pred = self.model(img)
            pred = pred[0]
            pred = pred.clone().cpu().detach()
            pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=1000)

            if self.model_name == 'yolov5s.pt':
                car_cls_id = 2  # 获取 "car" 的类别ID
                filtered_pred = []
                for det in pred:
                    if len(det) == 0:
                        continue
                    # 仅保留类别为 "car" 的检测框
                    car_mask = det[:, 5] == car_cls_id
                    det_car = det[car_mask]
                    filtered_pred.append(det_car)
                pred = filtered_pred     

            annotator = Annotator(ori_img, line_width=2)
            names = list(self.model.names.values())
            for i, det in enumerate(pred):
                if len(det):
                    det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], (ori_img.shape[:2])).round()
                    for *xyxy, conf, cls in reversed(det):
                        # ori_img = cv2.rectangle(ori_img, (int(xyxy[0].numpy()), int(xyxy[1].numpy())), (int(xyxy[2].numpy()), int(xyxy[3].numpy())), (0, 255, 0), 2)
                        # cv2.putText(ori_img, 'car', (int(xyxy[0].numpy()), int(xyxy[1].numpy() - 5)), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 1)
                        # if names[int(cls)]=="car":
                        annotator.box_label(xyxy, label=names[int(cls)], color=colors(int(cls)))
            
            if self.model_name == 'yolov5s.pt':
                if len(pred) > 0 and len(pred[0]) > 0:
                    boxes_xyxy = pred[0]  # 提取坐标 [x1,y1,x2,y2]
                else:
                    boxes_xyxy = torch.zeros((0, 6))  # 空张量
            else:
                boxes_xyxy = pred[0]

            boxes_xywh = xyxy2xywh(boxes_xyxy)

        elif 'v8' in self.model_name:
            result = self.model.predict(ori_img, verbose=False)[0]
            cls = result.boxes.cls.cpu()
            conf = result.boxes.conf.cpu()
            xyxy = result.boxes.xyxy.cpu()
            xywh = result.boxes.xywh.cpu()
            boxes_xyxy = torch.cat((xyxy, conf.view(-1, 1), cls.view(-1, 1)), dim=1)
            boxes_xywh = torch.cat((xywh, conf.view(-1, 1), cls.view(-1, 1)), dim=1)
            mask = cls == 0
            boxes_xyxy = boxes_xyxy[mask]
            boxes_xywh = boxes_xywh[mask]
            clss = cls[mask].tolist()
            names = result.names

            annotator = Annotator(ori_img, line_width=2)

            for box, cls in zip(boxes_xyxy, clss):
                annotator.box_label(box, label=names[int(cls)], color=colors(int(cls)))

        return ori_img, boxes_xyxy, boxes_xywh


if __name__ == '__main__':
    detector = VehicleDetector()
    img_path = os.path.join(root_path, 'example.jpg')
    img = cv2.imread(img_path)
    img = detector.get_vehicles(img)
    cv2.imwrite(os.path.join(root_path, 'result.jpg'), img)
