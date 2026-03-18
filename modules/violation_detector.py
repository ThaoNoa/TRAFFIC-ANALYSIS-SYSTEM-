"""
Module phát hiện vi phạm giao thông và tai nạn
"""

import cv2
import numpy as np
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ViolationDetector:
    def __init__(self):
        self.violation_zones = {}
        self.red_light_zones = []
        # Ngưỡng phát hiện tai nạn
        self.accel_threshold = -4.5  # m/s^2
        self.pose_angle_threshold = 35  # độ
        self.accident_min_frames = 25
        self.accident_counters = {}  # track_id -> số frame liên tiếp thỏa mãn
        logger.info("ViolationDetector initialized")

    def detect(self, frame, detections, tracks, ipm=None, pose_results=None):
        """
        Phát hiện vi phạm và tai nạn.
        Args:
            frame: ảnh đầu vào
            detections: list các detection từ detector
            tracks: list các track từ tracker
            ipm: IPMTransformer instance (để tính gia tốc)
            pose_results: dict track_id -> {'angle': góc nghiêng, 'confidence': ...}
        Returns:
            violations: list các vi phạm
        """
        violations = []
        current_time = datetime.now().strftime('%H:%M:%S')

        # Demo: phát hiện xe máy đi vào làn ô tô
        frame_center = frame.shape[1] // 2
        for track in tracks:
            track_id = track['track_id']
            x1, y1, x2, y2 = track['bbox']

            for det in detections:
                if self._is_same_object(det['bbox'], [x1, y1, x2, y2]):
                    if det['class_name'] == 'xe_may' and x1 > frame_center + 50:
                        violations.append({
                            'time': current_time,
                            'type': 'SAI_LAN',
                            'description': f'Xe máy ID {track_id} đi vào làn ô tô',
                            'track_id': track_id,
                            'bbox': [x1, y1, x2, y2]
                        })
                    break

        # Phát hiện tai nạn nếu có IPM và pose
        if ipm is not None and pose_results is not None:
            for track in tracks:
                track_id = track['track_id']
                if track_id not in pose_results:
                    continue
                pose = pose_results[track_id]
                angle = pose.get('angle')
                if angle is None:
                    continue

                # Tính gia tốc từ lịch sử track (cần lưu history trong tracker)
                # Ở đây giả sử track có history speeds
                speeds = track.get('speed_history', [])
                if len(speeds) >= 2:
                    accel = speeds[-1] - speeds[-2]  # delta v / 1 frame? Cần dt
                    # Thực tế cần dt, tạm tính đơn giản
                    if accel < self.accel_threshold and angle < self.pose_angle_threshold:
                        # Tăng counter
                        self.accident_counters[track_id] = self.accident_counters.get(track_id, 0) + 1
                        if self.accident_counters[track_id] >= self.accident_min_frames:
                            violations.append({
                                'time': current_time,
                                'type': 'ACCIDENT',
                                'description': f'Phát hiện tai nạn/ngã xe ID {track_id}',
                                'track_id': track_id,
                                'bbox': [x1, y1, x2, y2]
                            })
                            # Reset counter để không báo lại liên tục
                            self.accident_counters[track_id] = 0
                    else:
                        self.accident_counters[track_id] = 0

        return violations

    def _is_same_object(self, bbox1, bbox2, iou_threshold=0.5):
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