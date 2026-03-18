"""
Module chuyển đổi không gian IPM (Bird's Eye View)
Tính vận tốc và gia tốc từ tọa độ bounding box
"""

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

class IPMTransformer:
    def __init__(self, src_points, dst_points, frame_shape):
        """
        Args:
            src_points: 4 điểm trong ảnh gốc (góc nhìn phối cảnh) – list of (x,y)
            dst_points: 4 điểm tương ứng trong ảnh bird's eye view (tỷ lệ mét)
            frame_shape: (h,w) của frame gốc
        """
        self.src_points = np.float32(src_points)
        self.dst_points = np.float32(dst_points)
        self.h, self.w = frame_shape[:2]
        self.H = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
        self.H_inv = cv2.getPerspectiveTransform(self.dst_points, self.src_points)
        self.scale_x = (dst_points[1][0] - dst_points[0][0]) / (src_points[1][0] - src_points[0][0])  # pixel -> mét
        self.scale_y = (dst_points[2][1] - dst_points[0][1]) / (src_points[2][1] - src_points[0][1])
        logger.info(f"IPM initialized with scale: x={self.scale_x:.3f} m/pixel, y={self.scale_y:.3f} m/pixel")

    def warp(self, frame):
        """Chuyển toàn bộ frame sang bird's eye view (dùng để debug)"""
        return cv2.warpPerspective(frame, self.H, (self.w, self.h))

    def image_to_world(self, x, y):
        """Chuyển tọa độ ảnh (pixel) sang tọa độ thế giới (mét)"""
        pts = np.float32([[[x, y]]])
        world = cv2.perspectiveTransform(pts, self.H)[0][0]
        return world[0], world[1]

    def world_to_image(self, X, Y):
        """Chuyển tọa độ thế giới (mét) sang ảnh (pixel)"""
        pts = np.float32([[[X, Y]]])
        img = cv2.perspectiveTransform(pts, self.H_inv)[0][0]
        return int(img[0]), int(img[1])

    def compute_speed(self, track_history, fps):
        """
        Tính vận tốc (m/s) từ lịch sử track.
        Args:
            track_history: list các dict {'bbox': [x1,y1,x2,y2], 'time': frame_idx}
            fps: số frame/giây
        Returns:
            speed_mps: vận tốc trung bình (m/s) hoặc None nếu không đủ dữ liệu
        """
        if len(track_history) < 2:
            return None
        # Lấy tọa độ đáy bounding box (giữa theo x, đáy y)
        x1, y1, x2, y2 = track_history[-1]['bbox']
        bottom_x = (x1 + x2) / 2
        bottom_y = y2
        world_x, world_y = self.image_to_world(bottom_x, bottom_y)

        x1_prev, y1_prev, x2_prev, y2_prev = track_history[-2]['bbox']
        bottom_x_prev = (x1_prev + x2_prev) / 2
        bottom_y_prev = y2_prev
        world_x_prev, world_y_prev = self.image_to_world(bottom_x_prev, bottom_y_prev)

        # Khoảng cách di chuyển (mét)
        dist = np.hypot(world_x - world_x_prev, world_y - world_y_prev)
        dt = (track_history[-1]['time'] - track_history[-2]['time']) / fps  # giây
        if dt <= 0:
            return None
        return dist / dt

    def compute_acceleration(self, speed_history, dt=1.0):
        """Tính gia tốc (m/s^2) từ lịch sử vận tốc"""
        if len(speed_history) < 2:
            return None
        return (speed_history[-1] - speed_history[-2]) / dt