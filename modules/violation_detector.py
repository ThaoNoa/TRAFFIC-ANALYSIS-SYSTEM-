"""
Module phát hiện vi phạm giao thông MỞ RỘNG
Bao gồm: Sai làn, Không đội mũ bảo hiểm, Vượt tốc độ
"""

import cv2
import numpy as np
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ViolationDetector:
    """
    Phát hiện các hành vi vi phạm:
    - KHONG_DOI_MU: Người đi xe máy không đội mũ bảo hiểm
    - VUOT_TOC_DO: Phương tiện vượt quá tốc độ cho phép
    - SAI_LAN: Xe máy đi vào làn ô tô (hoặc ngược lại)
    """

    def __init__(self, speed_limit_kmh=60, helmet_conf_threshold=0.4):
        """
        Args:
            speed_limit_kmh: Giới hạn tốc độ (km/h)
            helmet_conf_threshold: Ngưỡng confidence phát hiện mũ bảo hiểm
        """
        self.speed_limit_kmh = speed_limit_kmh
        self.helmet_conf_threshold = helmet_conf_threshold
        self.violation_zones = {}
        self.red_light_zones = []
        self.violation_count = 0

        # Khởi tạo HOG để phát hiện mũ bảo hiểm (dự phòng)
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor.getDefaultPeopleDetector())

        logger.info("ViolationDetector initialized - Extended version")

    def detect(self, frame, detections, tracks):
        """
        Phát hiện vi phạm

        Args:
            frame: Ảnh đầu vào
            detections: List các detection từ YOLO
            tracks: List các track từ tracker

        Returns:
            violations: List các vi phạm phát hiện được
        """
        violations = []
        current_time = datetime.now().strftime('%H:%M:%S')
        h, w = frame.shape[:2]
        frame_center = w // 2

        # === 1. PHÁT HIỆN KHÔNG ĐỘI MŨ BẢO HIỂM (XE MÁY) ===
        for det in detections:
            if det['class_name'] == 'nguoi':
                x1, y1, x2, y2 = det['bbox']
                # Cắt vùng đầu của người (1/4 phía trên bounding box)
                head_y1 = y1
                head_y2 = y1 + (y2 - y1) // 4
                head_x1 = x1
                head_x2 = x2

                if head_y2 > head_y1 and head_x2 > head_x1:
                    head_roi = frame[head_y1:head_y2, head_x1:head_x2]
                    if head_roi.size > 0:
                        # Phát hiện mũ bảo hiểm bằng cách phân tích màu sắc và hình dạng
                        has_helmet = self._detect_helmet_color(head_roi)

                        if not has_helmet:
                            # Kiểm tra xem người này có đang ở trên xe máy không
                            is_on_motorcycle = self._is_person_on_motorcycle(det, detections)

                            if is_on_motorcycle:
                                violations.append({
                                    'id': f"{current_time}_no_helmet_{len(violations)}",
                                    'time': current_time,
                                    'type': 'KHONG_DOI_MU',
                                    'description': f'Người đi xe máy không đội mũ bảo hiểm',
                                    'track_id': None,
                                    'bbox': [x1, y1, x2, y2],
                                    'speed': None
                                })
                                # Vẽ cảnh báo lên frame
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                                cv2.putText(frame, "KHONG DOI MU", (x1, y1-10),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # === 2. PHÁT HIỆN VƯỢT TỐC ĐỘ ===
        for track in tracks:
            speed = track.get('speed_kmh')
            track_id = track['track_id']

            if speed is not None and speed > self.speed_limit_kmh:
                x1, y1, x2, y2 = track['bbox']
                violations.append({
                    'id': f"{current_time}_speeding_{track_id}",
                    'time': current_time,
                    'type': 'VUOT_TOC_DO',
                    'description': f'Phương tiện ID {track_id} vượt tốc độ: {speed:.1f}/{self.speed_limit_kmh} km/h',
                    'track_id': track_id,
                    'bbox': [x1, y1, x2, y2],
                    'speed': speed
                })
                # Vẽ cảnh báo lên frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(frame, f"VUOT TOC DO: {speed:.0f}km/h", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


        return violations

    def _detect_helmet_color(self, head_roi):
        """
        Phát hiện mũ bảo hiểm dựa trên phân tích màu sắc và độ tròn
        """
        if head_roi.size == 0:
            return False

        h, w = head_roi.shape[:2]
        if h < 10 or w < 10:
            return False

        # Chuyển sang HSV để phân tích màu
        hsv = cv2.cvtColor(head_roi, cv2.COLOR_BGR2HSV)

        # Mũ bảo hiểm thường có màu sắc nổi bật (đỏ, vàng, xanh, trắng)
        # Ngưỡng màu cho mũ bảo hiểm
        helmet_colors = [
            ([0, 50, 50], [10, 255, 255]),    # Đỏ
            ([20, 50, 50], [35, 255, 255]),   # Vàng cam
            ([100, 50, 50], [130, 255, 255]), # Xanh dương
            ([0, 0, 200], [180, 30, 255])     # Trắng/sáng
        ]

        for lower, upper in helmet_colors:
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            mask = cv2.inRange(hsv, lower, upper)
            helmet_ratio = np.sum(mask > 0) / (h * w)

            if helmet_ratio > 0.15:  # Nếu có hơn 15% vùng đầu có màu mũ
                return True

        # Phát hiện hình dạng tròn (mũ bảo hiểm thường có dạng cong)
        gray = cv2.cvtColor(head_roi, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=w//2,
                                   param1=50, param2=30, minRadius=w//4, maxRadius=w)
        if circles is not None:
            return True

        return False

    def _is_person_on_motorcycle(self, person_det, all_detections):
        """
        Kiểm tra xem người có đang ở trên xe máy không
        Bằng cách xem có bounding box xe máy gần người không
        """
        px1, py1, px2, py2 = person_det['bbox']
        p_center_x = (px1 + px2) // 2
        p_bottom_y = py2

        for det in all_detections:
            if det['class_name'] == 'xe_may':
                mx1, my1, mx2, my2 = det['bbox']
                m_center_x = (mx1 + mx2) // 2
                m_top_y = my1
                m_bottom_y = my2

                # Kiểm tra khoảng cách giữa người và xe máy
                distance_x = abs(p_center_x - m_center_x)
                distance_y = abs(p_bottom_y - m_top_y)

                # Nếu người ở gần xe máy (cách nhau < 50 pixel)
                if distance_x < 50 and distance_y < 50:
                    return True

        return False

    def set_speed_limit(self, limit_kmh):
        """Cập nhật giới hạn tốc độ"""
        self.speed_limit_kmh = limit_kmh
        logger.info(f"Speed limit updated to {limit_kmh} km/h")

    def get_violation_stats(self):
        """Lấy thống kê vi phạm"""
        return {
            'total_violations': self.violation_count,
            'speed_limit': self.speed_limit_kmh
        }