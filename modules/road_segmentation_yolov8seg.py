"""
Module phân đoạn mặt đường sử dụng YOLOv8-Seg (đã fine-tune)
Nhận diện đồng thời bounding box phương tiện và mask mặt đường (3 lớp)
"""

import cv2
import numpy as np
import logging
from ultralytics import YOLO

logger = logging.getLogger(__name__)

class RoadSegmentorYOLOSeg:
    def __init__(self, model_path='yolov8n-seg.pt', conf_threshold=0.5, fallback=None):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.fallback = fallback
        logger.info(f"YOLOv8-Seg RoadSegmentor initialized with model {model_path}")

        # Ánh xạ lớp: 0=background, 1=pothole, 2=crack (tuỳ chỉnh theo dataset)
        self.class_names = {0: 'background', 1: 'pothole', 2: 'crack'}

    def segment(self, frame, use_fallback=False):
        h, w = frame.shape[:2]

        if use_fallback and self.fallback:
            road_mask, sidewalk_mask, roi_coords, _ = self.fallback.extract_road_and_sidewalk(frame)
            return road_mask, None, None, roi_coords

        # Dùng YOLOv8-seg
        results = self.model(frame, conf=self.conf_threshold, verbose=False)[0]

        road_mask = np.zeros((h, w), dtype=np.uint8)
        pothole_mask = np.zeros((h, w), dtype=np.uint8)
        crack_mask = np.zeros((h, w), dtype=np.uint8)

        if results.masks is not None:
            for mask, cls in zip(results.masks.data, results.boxes.cls):
                cls_id = int(cls)
                if cls_id not in self.class_names:
                    continue
                # mask là tensor (N, H, W) - chuyển về numpy và resize
                mask_np = mask.cpu().numpy().astype(np.uint8)
                # Resize về kích thước frame gốc
                mask_np = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
                if cls_id == 1:  # ổ gà
                    pothole_mask = cv2.bitwise_or(pothole_mask, mask_np * 255)
                elif cls_id == 2:  # vết nứt
                    crack_mask = cv2.bitwise_or(crack_mask, mask_np * 255)
                else:  # background (không dùng)
                    pass

        # Tạo road_mask tổng hợp: các pixel không phải background
        road_mask = cv2.bitwise_or(pothole_mask, crack_mask)

        # Nếu road_mask rỗng, thử lấy toàn bộ khung hình (fallback)
        if np.sum(road_mask) == 0:
            logger.warning("YOLOv8-Seg không tìm thấy mặt đường, dùng toàn bộ frame làm ROI")
            road_mask = np.ones((h, w), dtype=np.uint8) * 255
            roi_coords = [0, h, 0, w]
        else:
            # Tìm contour lớn nhất để lấy ROI
            contours, _ = cv2.findContours(road_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                main_contour = max(contours, key=cv2.contourArea)
                x, y, w_road, h_road = cv2.boundingRect(main_contour)
                roi_coords = [y, y+h_road, x, x+w_road]
            else:
                roi_coords = [0, h, 0, w]

        return road_mask, pothole_mask, crack_mask, roi_coords