"""
Module Inverse Perspective Mapping (IPM)
Chuyển đổi tọa độ từ camera 2D sang tọa độ mặt đường thực tế
Dùng để tính vận tốc (km/h) và gia tốc
"""

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


class IPMTransformer:
    """
    Inverse Perspective Mapping Transformer
    Chuyển đổi điểm từ ảnh (2D) sang mặt đường (3D)
    """

    def __init__(self,
                 frame_width=640,
                 frame_height=480,
                 camera_height=1.2,  # Chiều cao camera (mét)
                 camera_pitch=15,  # Góc nghiêng camera (độ)
                 camera_fov=60,  # Field of view (độ)
                 road_width_meters=8.0):  # Chiều rộng đường (mét)
        """
        Khởi tạo IPM Transformer

        Args:
            frame_width: Chiều rộng frame (pixel)
            frame_height: Chiều cao frame (pixel)
            camera_height: Chiều cao camera so với mặt đường (m)
            camera_pitch: Góc nghiêng của camera so với phương ngang (độ)
            camera_fov: Góc mở của camera (độ)
            road_width_meters: Chiều rộng đường thực tế (m)
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.camera_height = camera_height
        self.camera_pitch = np.radians(camera_pitch)
        self.camera_fov = np.radians(camera_fov)
        self.road_width_meters = road_width_meters

        # Tính toán ma trận chuyển đổi
        self._compute_transform_matrices()

        logger.info(f"IPMTransformer initialized - height={camera_height}m, pitch={camera_pitch}°")

    def _compute_transform_matrices(self):
        """Tính ma trận chuyển đổi IPM"""
        # Điểm nguồn (4 góc vùng quan tâm trong ảnh)
        src_points = np.float32([
            [self.frame_width * 0.2, self.frame_height * 0.6],  # Trên-trái
            [self.frame_width * 0.8, self.frame_height * 0.6],  # Trên-phải
            [self.frame_width * 0.95, self.frame_height],  # Dưới-phải
            [self.frame_width * 0.05, self.frame_height]  # Dưới-trái
        ])

        # Điểm đích (trong mặt phẳng đường, đơn vị mét)
        # Giả sử vùng quan tâm dài 30m, rộng bằng đường
        road_length_meters = 30.0
        dst_points = np.float32([
            [0, 0],
            [self.road_width_meters, 0],
            [self.road_width_meters, road_length_meters],
            [0, road_length_meters]
        ])

        # Tính ma trận perspective transform
        self.M = cv2.getPerspectiveTransform(src_points, dst_points)
        self.M_inv = cv2.getPerspectiveTransform(dst_points, src_points)

        # Tỷ lệ pixel/mét
        self.pixels_per_meter_x = self.frame_width / self.road_width_meters
        self.pixels_per_meter_y = self.frame_height / road_length_meters

    def image_to_road(self, x, y):
        """
        Chuyển đổi điểm từ ảnh sang tọa độ mặt đường (mét)

        Args:
            x, y: Tọa độ pixel trong ảnh

        Returns:
            (road_x, road_y): Tọa độ trên mặt đường (mét)
        """
        # Chuyển sang tọa độ thuần nhất
        src_point = np.array([[[x, y]]], dtype=np.float32)
        dst_point = cv2.perspectiveTransform(src_point, self.M)

        if dst_point is not None and len(dst_point) > 0:
            return dst_point[0][0][0], dst_point[0][0][1]
        return 0, 0

    def road_to_image(self, road_x, road_y):
        """
        Chuyển đổi từ tọa độ mặt đường sang ảnh

        Args:
            road_x, road_y: Tọa độ trên mặt đường (mét)

        Returns:
            (x, y): Tọa độ pixel trong ảnh
        """
        src_point = np.array([[[road_x, road_y]]], dtype=np.float32)
        dst_point = cv2.perspectiveTransform(src_point, self.M_inv)

        if dst_point is not None and len(dst_point) > 0:
            return int(dst_point[0][0][0]), int(dst_point[0][0][1])
        return 0, 0

    def compute_velocity(self, prev_center, curr_center, time_delta):
        """
        Tính vận tốc từ 2 vị trí liên tiếp

        Args:
            prev_center: Tọa độ trước đó (x, y) trong ảnh
            curr_center: Tọa độ hiện tại (x, y) trong ảnh
            time_delta: Thời gian giữa 2 frame (giây)

        Returns:
            velocity_kmh: Vận tốc (km/h)
            velocity_ms: Vận tốc (m/s)
        """
        # Chuyển sang tọa độ thực tế
        prev_x, prev_y = self.image_to_road(prev_center[0], prev_center[1])
        curr_x, curr_y = self.image_to_road(curr_center[0], curr_center[1])

        # Tính khoảng cách di chuyển (mét)
        distance = np.sqrt((curr_x - prev_x) ** 2 + (curr_y - prev_y) ** 2)

        # Tính vận tốc
        if time_delta > 0:
            velocity_ms = distance / time_delta
            velocity_kmh = velocity_ms * 3.6
            return velocity_kmh, velocity_ms
        return 0, 0

    def compute_acceleration(self, prev_vel, curr_vel, time_delta):
        """
        Tính gia tốc từ 2 vận tốc liên tiếp

        Args:
            prev_vel: Vận tốc trước đó (m/s)
            curr_vel: Vận tốc hiện tại (m/s)
            time_delta: Thời gian giữa 2 lần đo (giây)

        Returns:
            acceleration: Gia tốc (m/s²)
        """
        if time_delta > 0:
            return (curr_vel - prev_vel) / time_delta
        return 0

    def draw_bird_eye_view(self, frame, road_mask=None):
        """
        Vẽ góc nhìn từ trên xuống (Bird Eye View)

        Args:
            frame: Ảnh gốc
            road_mask: Mask mặt đường (tùy chọn)

        Returns:
            bev: Ảnh góc nhìn từ trên xuống
        """
        h, w = frame.shape[:2]

        # Tạo vùng quan tâm
        src_points = np.float32([
            [w * 0.2, h * 0.6],
            [w * 0.8, h * 0.6],
            [w * 0.95, h],
            [w * 0.05, h]
        ])

        # Kích thước ảnh BEV
        bev_width = int(self.road_width_meters * self.pixels_per_meter_x)
        bev_height = 600

        dst_points = np.float32([
            [0, 0],
            [bev_width, 0],
            [bev_width, bev_height],
            [0, bev_height]
        ])

        M_bev = cv2.getPerspectiveTransform(src_points, dst_points)
        bev = cv2.warpPerspective(frame, M_bev, (bev_width, bev_height))

        return bev


class VehicleTrackerWithIPM:
    """
    Kết hợp tracking và IPM để tính vận tốc, gia tốc
    """

    def __init__(self, ipm_transformer):
        self.ipm = ipm_transformer
        self.vehicle_history = {}  # track_id -> list of (timestamp, position, velocity)

    def update_vehicle(self, track_id, center, timestamp):
        """
        Cập nhật vị trí phương tiện và tính toán động học

        Args:
            track_id: ID của phương tiện
            center: Tâm bounding box (x, y) trong ảnh
            timestamp: Thời gian (giây)

        Returns:
            velocity_kmh: Vận tốc (km/h)
            acceleration: Gia tốc (m/s²)
        """
        # Chuyển sang tọa độ thực tế
        road_x, road_y = self.ipm.image_to_road(center[0], center[1])

        # Khởi tạo lịch sử nếu chưa có
        if track_id not in self.vehicle_history:
            self.vehicle_history[track_id] = []

        history = self.vehicle_history[track_id]
        history.append({
            'timestamp': timestamp,
            'center_pixel': center,
            'center_road': (road_x, road_y),
            'velocity': 0,
            'acceleration': 0
        })

        # Giữ tối đa 30 frame lịch sử
        if len(history) > 30:
            history.pop(0)

        # Tính vận tốc và gia tốc nếu có đủ dữ liệu
        velocity_kmh = 0
        acceleration = 0

        if len(history) >= 2:
            prev = history[-2]
            curr = history[-1]
            time_delta = curr['timestamp'] - prev['timestamp']

            # Tính vận tốc
            velocity_kmh, velocity_ms = self.ipm.compute_velocity(
                prev['center_pixel'], curr['center_pixel'], time_delta
            )
            history[-1]['velocity'] = velocity_ms

            # Tính gia tốc
            if len(history) >= 3:
                prev_vel = history[-2]['velocity']
                curr_vel = history[-1]['velocity']
                acceleration = self.ipm.compute_acceleration(prev_vel, curr_vel, time_delta)
                history[-1]['acceleration'] = acceleration

        return velocity_kmh, acceleration

    def get_speed(self, track_id):
        """Lấy vận tốc hiện tại của phương tiện (km/h)"""
        if track_id in self.vehicle_history and self.vehicle_history[track_id]:
            last = self.vehicle_history[track_id][-1]
            return last['velocity'] * 3.6  # m/s -> km/h
        return 0

    def is_speeding(self, track_id, speed_limit_kmh=60):
        """Kiểm tra có đang vượt quá tốc độ không"""
        speed = self.get_speed(track_id)
        return speed > speed_limit_kmh