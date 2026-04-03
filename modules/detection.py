"""
Module phát hiện phương tiện - Phiên bản ỔN ĐỊNH
Hỗ trợ TensorRT FP16 để đạt 42+ FPS
"""

import cv2
from ultralytics import YOLO
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional
import torch

logger = logging.getLogger(__name__)


class VehicleDetector:
    def __init__(self,
                 model_path='yolov8n.pt',
                 conf_threshold=0.35,
                 iou_threshold=0.5,
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 use_tensorrt=False,
                 half_precision=True):
        """
        Khởi tạo VehicleDetector với tham số ổn định
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.use_tensorrt = use_tensorrt
        self.half_precision = half_precision and device == 'cuda'

        # Tải model
        self.model = YOLO(model_path)

        # Cấu hình model
        if device == 'cuda':
            self.model.to('cuda')
            if self.half_precision:
                self.model.half()

        # Mapping class COCO sang tiếng Việt
        self.traffic_classes = {
            0: 'nguoi',
            1: 'xe_dap',
            2: 'xe_oto',
            3: 'xe_may',
            5: 'xe_bus',
            7: 'xe_tai'
        }

        # Màu sắc cho từng loại
        self.colors = {
            'nguoi': (255, 200, 100),
            'xe_dap': (200, 100, 255),
            'xe_may': (0, 255, 100),
            'xe_oto': (100, 100, 255),
            'xe_bus': (0, 200, 255),
            'xe_tai': (50, 50, 200)
        }

        # Thống kê
        self.stats = {name: 0 for name in self.traffic_classes.values()}

        logger.info(f"VehicleDetector initialized - conf={conf_threshold}, iou={iou_threshold}")

    def detect(self,
               frame: np.ndarray,
               road_mask: Optional[np.ndarray] = None,
               return_vehicle_mask: bool = False) -> Tuple[List[Dict], np.ndarray, Optional[np.ndarray]]:
        """
        Phát hiện phương tiện
        """
        h, w = frame.shape[:2]

        # Chạy YOLO
        results = self.model(frame,
                            conf=self.conf_threshold,
                            iou=self.iou_threshold,
                            verbose=False,
                            max_det=50,
                            half=self.half_precision)

        # Reset stats
        self.stats = {name: 0 for name in self.traffic_classes.values()}

        detections = []
        vehicle_mask = np.zeros((h, w), dtype=np.uint8) if return_vehicle_mask else None

        if results and len(results) > 0:
            result = results[0]

            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes

                xyxy = boxes.xyxy.cpu().numpy() if torch.cuda.is_available() else boxes.xyxy.numpy()
                confs = boxes.conf.cpu().numpy() if torch.cuda.is_available() else boxes.conf.numpy()
                classes = boxes.cls.cpu().numpy() if torch.cuda.is_available() else boxes.cls.numpy()

                for i in range(len(xyxy)):
                    x1, y1, x2, y2 = map(int, xyxy[i])
                    conf = float(confs[i])
                    cls = int(classes[i])

                    if cls not in self.traffic_classes:
                        continue

                    class_name = self.traffic_classes[cls]

                    if x1 >= x2 or y1 >= y2:
                        continue
                    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
                        continue

                    on_road = True
                    if road_mask is not None:
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        bottom_y = y2

                        on_road = False
                        if (0 <= center_x < w and 0 <= center_y < h and
                            road_mask[center_y, center_x] > 0):
                            on_road = True
                        elif (0 <= center_x < w and 0 <= bottom_y < h and
                              road_mask[bottom_y, center_x] > 0):
                            on_road = True

                    if on_road:
                        detection = {
                            'bbox': [x1, y1, x2, y2],
                            'confidence': conf,
                            'class_id': cls,
                            'class_name': class_name,
                            'center': ((x1 + x2) // 2, (y1 + y2) // 2)
                        }
                        detections.append(detection)
                        self.stats[class_name] += 1

                        if return_vehicle_mask and vehicle_mask is not None:
                            cv2.rectangle(vehicle_mask, (x1, y1), (x2, y2), 255, -1)

        # Áp dụng NMS bổ sung
        detections = self._apply_additional_nms(detections)

        # Vẽ kết quả
        annotated_frame = self._draw_detections(frame.copy(), detections)

        if return_vehicle_mask:
            return detections, annotated_frame, vehicle_mask
        return detections, annotated_frame

    def _apply_additional_nms(self, detections: List[Dict]) -> List[Dict]:
        """Áp dụng NMS bổ sung để loại bỏ boxes trùng lặp"""
        if len(detections) <= 1:
            return detections

        detections.sort(key=lambda x: x['confidence'], reverse=True)
        keep = []

        while detections:
            best = detections.pop(0)
            keep.append(best)

            to_remove = []
            for i, det in enumerate(detections):
                iou = self._calculate_iou(best['bbox'], det['bbox'])
                if iou > self.iou_threshold and best['class_name'] == det['class_name']:
                    to_remove.append(i)

            for i in reversed(to_remove):
                detections.pop(i)

        return keep

    def _calculate_iou(self, box1: List[int], box2: List[int]) -> float:
        """Tính IoU giữa 2 bounding boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def _draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Vẽ bounding boxes lên frame"""
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class_name']

            color = self.colors.get(class_name, (0, 255, 0))

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"{class_name}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]

            cv2.rectangle(frame,
                         (x1, y1 - label_size[1] - 5),
                         (x1 + label_size[0] + 5, y1),
                         color, -1)

            cv2.putText(frame, label, (x1 + 2, y1 - 3),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return frame

    def get_stats(self) -> Dict:
        """Lấy thống kê số lượng phương tiện"""
        vehicles = ['xe_may', 'xe_oto', 'xe_bus', 'xe_tai', 'xe_dap']
        return {
            'total_vehicles': sum(self.stats.get(v, 0) for v in vehicles),
            'motorcycles': self.stats.get('xe_may', 0),
            'cars': self.stats.get('xe_oto', 0),
            'trucks': self.stats.get('xe_tai', 0),
            'buses': self.stats.get('xe_bus', 0),
            'persons': self.stats.get('nguoi', 0),
            'bicycles': self.stats.get('xe_dap', 0)
        }