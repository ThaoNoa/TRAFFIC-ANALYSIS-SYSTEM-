"""
Module phát hiện vi phạm giao thông
"""

import cv2
import numpy as np
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ViolationDetector:
    """
    Phát hiện các hành vi vi phạm: vượt đèn đỏ, đi sai làn, dừng đỗ sai
    """

    def __init__(self):
        self.violation_zones = {}  # Các vùng cấm
        self.red_light_zones = []  # Vùng đèn đỏ
        logger.info("ViolationDetector initialized")

    def detect(self, frame, detections, tracks):
        """
        Phát hiện vi phạm

        Args:
            frame: Ảnh đầu vào
            detections: List các detection
            tracks: List các track từ tracker

        Returns:
            violations: List các vi phạm phát hiện được
        """
        violations = []
        current_time = datetime.now().strftime('%H:%M:%S')

        # Demo: Phát hiện xe máy đi vào làn ô tô (giả định)
        frame_center = frame.shape[1] // 2

        for track in tracks:
            track_id = track['track_id']
            x1, y1, x2, y2 = track['bbox']

            # Tìm detection tương ứng
            for det in detections:
                if self._is_same_object(det['bbox'], [x1, y1, x2, y2]):
                    # Kiểm tra nếu xe máy đi vào làn ô tô
                    if det['class_name'] == 'xe_may' and x1 > frame_center + 50:
                        violations.append({
                            'time': current_time,
                            'type': 'SAI_LAN',
                            'description': f'Xe máy ID {track_id} đi vào làn ô tô',
                            'track_id': track_id,
                            'bbox': [x1, y1, x2, y2]
                        })
                    break

        return violations

    def _is_same_object(self, bbox1, bbox2, iou_threshold=0.5):
        """Kiểm tra xem hai bounding box có cùng object không"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection

        iou = intersection / union if union > 0 else 0
        return iou > iou_threshold