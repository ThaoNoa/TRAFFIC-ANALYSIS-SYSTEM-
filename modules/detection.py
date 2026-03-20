"""
Module phát hiện phương tiện - Phiên bản ỔN ĐỊNH
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
                 conf_threshold=0.35,      # Tăng ngưỡng lên để giảm false positive
                 iou_threshold=0.5,        # Giảm iou để NMS mạnh hơn
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Khởi tạo VehicleDetector với tham số ổn định

        Args:
            model_path: Đường dẫn model YOLO
            conf_threshold: Ngưỡng confidence (tăng lên để giảm nhiễu)
            iou_threshold: Ngưỡng IoU cho NMS
            device: Thiết bị tính toán
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device

        # Tải model
        self.model = YOLO(model_path)

        # Cấu hình model
        if device == 'cuda':
            self.model.to('cuda')

        # Mapping class COCO sang tiếng Việt
        self.traffic_classes = {
            0: 'nguoi',        # person
            1: 'xe_dap',       # bicycle
            2: 'xe_oto',       # car
            3: 'xe_may',       # motorcycle
            5: 'xe_bus',       # bus
            7: 'xe_tai'        # truck
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
        Phát hiện phương tiện - phiên bản ổn định

        Args:
            frame: Ảnh đầu vào (BGR)
            road_mask: Mask mặt đường (None nếu không có)
            return_vehicle_mask: Có trả về mask không

        Returns:
            detections: Danh sách detection đã lọc
            annotated_frame: Frame đã vẽ
            vehicle_mask: Mask phương tiện (nếu return_vehicle_mask=True)
        """
        h, w = frame.shape[:2]

        # Chạy YOLO với tham số đã set
        results = self.model(frame,
                            conf=self.conf_threshold,
                            iou=self.iou_threshold,
                            verbose=False,
                            max_det=50)  # Giới hạn số lượng detection tối đa

        # Reset stats
        self.stats = {name: 0 for name in self.traffic_classes.values()}

        detections = []
        vehicle_mask = np.zeros((h, w), dtype=np.uint8) if return_vehicle_mask else None

        # Lấy kết quả từ YOLO
        if results and len(results) > 0:
            result = results[0]

            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes

                # Lấy tất cả thông tin
                xyxy = boxes.xyxy.cpu().numpy() if torch.cuda.is_available() else boxes.xyxy.numpy()
                confs = boxes.conf.cpu().numpy() if torch.cuda.is_available() else boxes.conf.numpy()
                classes = boxes.cls.cpu().numpy() if torch.cuda.is_available() else boxes.cls.numpy()

                # Lọc từng box
                for i in range(len(xyxy)):
                    x1, y1, x2, y2 = map(int, xyxy[i])
                    conf = float(confs[i])
                    cls = int(classes[i])

                    # Chỉ lấy các class quan tâm
                    if cls not in self.traffic_classes:
                        continue

                    class_name = self.traffic_classes[cls]

                    # Kiểm tra bounding box hợp lệ
                    if x1 >= x2 or y1 >= y2:
                        continue

                    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
                        continue

                    # Kiểm tra nằm trên đường (nếu có road_mask)
                    on_road = True
                    if road_mask is not None:
                        # Kiểm tra tâm và đáy của bounding box
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        bottom_y = y2

                        # Điều kiện: tâm và đáy phải trên đường
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

                        # Vẽ mask nếu cần
                        if return_vehicle_mask and vehicle_mask is not None:
                            cv2.rectangle(vehicle_mask, (x1, y1), (x2, y2), 255, -1)

        # Áp dụng NMS bổ sung để loại bỏ boxes chồng chéo
        detections = self._apply_additional_nms(detections)

        # Vẽ kết quả
        annotated_frame = self._draw_detections(frame.copy(), detections)

        if return_vehicle_mask:
            return detections, annotated_frame, vehicle_mask
        return detections, annotated_frame

    def _apply_additional_nms(self, detections: List[Dict]) -> List[Dict]:
        """
        Áp dụng NMS bổ sung để loại bỏ boxes trùng lặp
        """
        if len(detections) <= 1:
            return detections

        # Sắp xếp theo confidence giảm dần
        detections.sort(key=lambda x: x['confidence'], reverse=True)

        keep = []

        while detections:
            # Lấy box có confidence cao nhất
            best = detections.pop(0)
            keep.append(best)

            # Loại bỏ các box trùng lặp
            to_remove = []
            for i, det in enumerate(detections):
                iou = self._calculate_iou(best['bbox'], det['bbox'])
                # Nếu IoU cao và cùng class, loại bỏ
                if iou > self.iou_threshold and best['class_name'] == det['class_name']:
                    to_remove.append(i)

            # Xóa các box đã đánh dấu (từ cuối lên để tránh lỗi index)
            for i in reversed(to_remove):
                detections.pop(i)

        return keep

    def _calculate_iou(self, box1: List[int], box2: List[int]) -> float:
        """
        Tính IoU giữa 2 bounding boxes
        """
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
        """
        Vẽ bounding boxes lên frame (đơn giản và ổn định)
        """
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class_name']
            conf = det['confidence']

            color = self.colors.get(class_name, (0, 255, 0))

            # Vẽ bounding box với độ dày cố định
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Vẽ label
            label = f"{class_name}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]

            # Vẽ background cho label
            cv2.rectangle(frame,
                         (x1, y1 - label_size[1] - 5),
                         (x1 + label_size[0] + 5, y1),
                         color, -1)

            # Vẽ text
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

    def update_thresholds(self, conf_threshold: float = None, iou_threshold: float = None):
        """Cập nhật ngưỡng detection"""
        if conf_threshold is not None:
            self.conf_threshold = conf_threshold
            logger.info(f"Updated conf_threshold to {conf_threshold}")

        if iou_threshold is not None:
            self.iou_threshold = iou_threshold
            logger.info(f"Updated iou_threshold to {iou_threshold}")


# Phiên bản đơn giản hơn - dùng trực tiếp YOLO mà không xử lý phức tạp
class SimpleVehicleDetector:
    """
    Phiên bản đơn giản nhất - chỉ dùng YOLO với tham số tối ưu
    """
    def __init__(self, model_path='yolov8n.pt', conf_threshold=0.4):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

        # Class mapping
        self.class_names = {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle',
            5: 'bus', 7: 'truck'
        }

        self.viet_names = {
            'person': 'nguoi', 'bicycle': 'xe_dap', 'car': 'xe_oto',
            'motorcycle': 'xe_may', 'bus': 'xe_bus', 'truck': 'xe_tai'
        }

    def detect(self, frame):
        """
        Detect và trả về kết quả đơn giản
        """
        results = self.model(frame, conf=self.conf_threshold, verbose=False)

        detections = []

        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                boxes = result.boxes

                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])

                    if cls in self.class_names:
                        class_en = self.class_names[cls]
                        class_vn = self.viet_names.get(class_en, class_en)

                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': conf,
                            'class': class_vn,
                            'class_en': class_en
                        })

        # Vẽ kết quả
        annotated = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label = f"{det['class']} {det['confidence']:.2f}"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return detections, annotated


# Test code
if __name__ == "__main__":
    # Sử dụng phiên bản ổn định
    detector = VehicleDetector(
        model_path='yolov8n.pt',
        conf_threshold=0.4,      # Cao hơn để giảm false positive
        iou_threshold=0.5        # NMS mạnh hơn
    )

    # Hoặc dùng phiên bản đơn giản hơn
    # detector = SimpleVehicleDetector(conf_threshold=0.45)

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect
        detections, annotated = detector.detect(frame)

        # Hiển thị số lượng
        if hasattr(detector, 'get_stats'):
            stats = detector.get_stats()
            cv2.putText(annotated, f"Vehicles: {stats['total_vehicles']}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Detection', annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()