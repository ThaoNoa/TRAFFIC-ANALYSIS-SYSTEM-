"""
Module phát hiện phương tiện
"""

import cv2
from ultralytics import YOLO
import logging
import numpy as np

logger = logging.getLogger(__name__)

class VehicleDetector:
    def __init__(self, model_path='yolov8n.pt', conf_threshold=0.5):
        self.conf_threshold = conf_threshold
        self.model = YOLO(model_path)

        # Định nghĩa các lớp
        self.traffic_classes = {
            0: 'nguoi', 1: 'xe_dap', 2: 'xe_may',
            3: 'xe_oto', 5: 'xe_bus', 7: 'xe_tai'
        }

        self.colors = {
            'nguoi': (255, 255, 0),
            'xe_dap': (255, 0, 255),
            'xe_may': (0, 255, 0),
            'xe_oto': (255, 0, 0),
            'xe_bus': (0, 165, 255),
            'xe_tai': (0, 0, 255)
        }

        self.stats = {name: 0 for name in self.traffic_classes.values()}
        logger.info("VehicleDetector initialized")

    def detect(self, frame, road_mask=None, return_vehicle_mask=False):
        """
        Phát hiện phương tiện, chỉ giữ xe trên mặt đường

        Args:
            frame: Ảnh đầu vào
            road_mask: Mask mặt đường
            return_vehicle_mask: Có trả về mask các phương tiện không

        Returns:
            detections: List detection đã lọc
            annotated_frame: Ảnh đã vẽ
            vehicle_mask: (optional) Mask các phương tiện
        """
        results = self.model(frame, conf=self.conf_threshold, verbose=False)[0]

        detections = []
        self.stats = {name: 0 for name in self.traffic_classes.values()}

        # Tạo vehicle mask nếu cần
        h, w = frame.shape[:2]
        vehicle_mask = np.zeros((h, w), dtype=np.uint8) if return_vehicle_mask else None

        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())

                if cls in self.traffic_classes:
                    class_name = self.traffic_classes[cls]

                    # Kiểm tra xe có nằm trên mặt đường không
                    on_road = True
                    if road_mask is not None and isinstance(road_mask, np.ndarray):
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2

                        h_mask, w_mask = road_mask.shape[:2]
                        if 0 <= center_x < w_mask and 0 <= center_y < h_mask:
                            if road_mask[center_y, center_x] == 0:
                                on_road = False

                    if on_road:
                        detection = {
                            'bbox': [x1, y1, x2, y2],
                            'confidence': conf,
                            'class_id': cls,
                            'class_name': class_name
                        }
                        detections.append(detection)
                        self.stats[class_name] += 1

                        # Thêm vào vehicle mask
                        if return_vehicle_mask and vehicle_mask is not None:
                            cv2.rectangle(vehicle_mask, (x1, y1), (x2, y2), 255, -1)

        # Vẽ kết quả
        annotated_frame = self.draw_detections(frame.copy(), detections)

        if return_vehicle_mask:
            return detections, annotated_frame, vehicle_mask
        else:
            return detections, annotated_frame

    def draw_detections(self, frame, detections):
        """Vẽ bounding box"""
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class_name']
            conf = det['confidence']

            color = self.colors.get(class_name, (0, 255, 0))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"{class_name}: {conf:.2f}"
            cv2.putText(frame, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame

    def get_stats(self):
        """Lấy thống kê"""
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