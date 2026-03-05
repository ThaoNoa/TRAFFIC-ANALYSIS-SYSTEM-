"""
Module phân tích tư thế đơn giản - Fallback khi MediaPipe không hoạt động
"""

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

class PoseAnalyzer:
    """
    Phiên bản đơn giản, chỉ dùng OpenCV để phát hiện người
    """

    def __init__(self):
        # Tạo HOG descriptor
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor.getDefaultPeopleDetector())
        logger.info("PoseAnalyzer initialized in simple mode with HOG")

    def analyze(self, frame, detections):
        """
        Phân tích đơn giản dùng HOG
        """
        annotated_frame = frame.copy()
        abnormal_poses = []

        # Phát hiện người bằng HOG
        (persons, _) = self.hog.detectMultiScale(
            frame,
            winStride=(4, 4),
            padding=(8, 8),
            scale=1.05
        )

        # Vẽ kết quả từ HOG
        for (x, y, w, h) in persons:
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(annotated_frame, "Person (HOG)", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Tính tỷ lệ khung hình
            aspect_ratio = h / w if w > 0 else 0

            # Phát hiện người nằm
            if aspect_ratio < 0.8 and h > 50:
                abnormal_poses.append({
                    'type': 'possible_fall',
                    'bbox': [x, y, x+w, y+h],
                    'aspect_ratio': aspect_ratio,
                    'method': 'HOG'
                })
                cv2.putText(annotated_frame, "FALL DETECTED!", (x, y-30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Cũng sử dụng detections từ YOLO
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det.get('confidence', 1.0)

            # Vẽ bounding box từ YOLO
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Person {conf:.2f}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return abnormal_poses, annotated_frame