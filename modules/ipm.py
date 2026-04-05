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
        self.road_length_meters = 30.0

        # Tính toán ma trận chuyển đổi
        self._compute_transform_matrices()

        logger.info(f"IPMTransformer initialized - height={camera_height}m, pitch={camera_pitch}°")

    def _compute_transform_matrices(self):
        """Tính ma trận chuyển đổi IPM (mặc định theo kích thước frame)."""
        road_length_meters = self.road_length_meters
        src_points = np.float32([
            [self.frame_width * 0.2, self.frame_height * 0.6],
            [self.frame_width * 0.8, self.frame_height * 0.6],
            [self.frame_width * 0.95, self.frame_height],
            [self.frame_width * 0.05, self.frame_height]
        ])
        dst_points = np.float32([
            [0, 0],
            [self.road_width_meters, 0],
            [self.road_width_meters, road_length_meters],
            [0, road_length_meters]
        ])
        self.M = cv2.getPerspectiveTransform(src_points, dst_points)
        self.M_inv = cv2.getPerspectiveTransform(dst_points, src_points)
        self.pixels_per_meter_x = self.frame_width / self.road_width_meters
        self.pixels_per_meter_y = self.frame_height / road_length_meters

    def set_frame_size(self, frame_width, frame_height):
        """Đồng bộ IPM với kích thước frame thực tế (ví dụ 640x480)."""
        self.frame_width = int(frame_width)
        self.frame_height = int(frame_height)
        self._compute_transform_matrices()

    def set_calibration_from_roi(self, y1, y2, x1, x2, frame_h, frame_w,
                                 road_length_meters=None):
        """
        Hiệu chỉnh homography theo ROI lòng đường (đã làm mượt) để IPM bám mặt đường ổn định hơn.
        roi_coords format: [y1, y2, x1, x2]
        """
        if road_length_meters is not None:
            self.road_length_meters = float(road_length_meters)
        fw, fh = int(frame_w), int(frame_h)
        x1 = int(np.clip(x1, 0, fw - 1))
        x2 = int(np.clip(x2, 0, fw - 1))
        y1 = int(np.clip(y1, 0, fh - 1))
        y2 = int(np.clip(y2, 0, fh - 1))
        if x2 <= x1 + 4 or y2 <= y1 + 4:
            return
        rw = float(x2 - x1)
        rh = float(y2 - y1)
        src_points = np.float32([
            [x1 + 0.08 * rw, y1 + 0.22 * rh],
            [x2 - 0.08 * rw, y1 + 0.22 * rh],
            [x2 - 0.02 * rw, y2],
            [x1 + 0.02 * rw, y2],
        ])
        L = self.road_length_meters
        W = self.road_width_meters
        dst_points = np.float32([
            [0, 0],
            [W, 0],
            [W, L],
            [0, L]
        ])
        self.M = cv2.getPerspectiveTransform(src_points, dst_points)
        self.M_inv = cv2.getPerspectiveTransform(dst_points, src_points)
        self.pixels_per_meter_x = fw / max(W, 1e-6)
        self.pixels_per_meter_y = fh / max(L, 1e-6)

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

        bev_width = max(160, int(self.road_width_meters * 25))
        bev_height = max(200, int(self.road_length_meters * 25))
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
    Kết hợp tracking và IPM để tính vận tốc, gia tốc (theo từng track_id).
    Dùng điểm chân (foot: giữa đáy bbox) trên mặt phẳng ảnh — ổn định hơn cho IPM.
    """

    def __init__(self, ipm_transformer, history_max=25, speed_ema_alpha=0.22,
                 max_speed_kmh=160.0, min_dt=1.0 / 120.0):
        self.ipm = ipm_transformer
        self.vehicle_history = {}
        self._speed_ema_kmh = {}
        self.history_max = history_max
        self.speed_ema_alpha = speed_ema_alpha
        self.max_speed_kmh = max_speed_kmh
        self.min_dt = min_dt

    def update_vehicle(self, track_id, foot_xy, timestamp):
        """
        Args:
            track_id: ID track
            foot_xy: (x, y) điểm giữa cạnh dưới bbox (tiếp xúc “mặt đường” trên ảnh)
            timestamp: time.time() (giây)

        Returns:
            (velocity_kmh_smoothed, acceleration_m_s2)
        """
        road_x, road_y = self.ipm.image_to_road(foot_xy[0], foot_xy[1])

        if track_id not in self.vehicle_history:
            self.vehicle_history[track_id] = []

        history = self.vehicle_history[track_id]
        history.append({
            'timestamp': timestamp,
            'center_pixel': (float(foot_xy[0]), float(foot_xy[1])),
            'center_road': (road_x, road_y),
            'velocity': 0.0,
            'acceleration': 0.0
        })

        if len(history) > self.history_max:
            history.pop(0)

        velocity_kmh_instant = 0.0
        acceleration = 0.0
        velocity_ms = 0.0

        if len(history) >= 2:
            prev = history[-2]
            curr = history[-1]
            time_delta = curr['timestamp'] - prev['timestamp']
            if time_delta < self.min_dt:
                time_delta = self.min_dt

            velocity_kmh_instant, velocity_ms = self.ipm.compute_velocity(
                prev['center_pixel'], curr['center_pixel'], time_delta
            )
            if not np.isfinite(velocity_kmh_instant):
                velocity_kmh_instant = 0.0
                velocity_ms = 0.0
            velocity_kmh_instant = float(np.clip(velocity_kmh_instant, 0.0, self.max_speed_kmh))
            history[-1]['velocity'] = velocity_ms

            if len(history) >= 3:
                prev_vel = history[-2]['velocity']
                curr_vel = history[-1]['velocity']
                acceleration = self.ipm.compute_acceleration(
                    prev_vel, curr_vel, time_delta
                )
                history[-1]['acceleration'] = acceleration

        # Làm mượt vận tốc hiển thị theo từng track_id
        if track_id not in self._speed_ema_kmh:
            self._speed_ema_kmh[track_id] = velocity_kmh_instant
        else:
            a = self.speed_ema_alpha
            self._speed_ema_kmh[track_id] = (
                a * velocity_kmh_instant + (1.0 - a) * self._speed_ema_kmh[track_id]
            )
        velocity_kmh_smoothed = float(np.clip(self._speed_ema_kmh[track_id], 0.0, self.max_speed_kmh))

        return velocity_kmh_smoothed, acceleration

    def prune(self, active_track_ids):
        """Xóa lịch sử track không còn xuất hiện (tránh rò bộ nhớ / ID cũ)."""
        active = set(active_track_ids)
        for tid in list(self.vehicle_history.keys()):
            if tid not in active:
                del self.vehicle_history[tid]
        for tid in list(self._speed_ema_kmh.keys()):
            if tid not in active:
                del self._speed_ema_kmh[tid]

    def reset(self):
        """Xóa toàn bộ lịch sử (khi mở video / phân tích mới)."""
        self.vehicle_history.clear()
        self._speed_ema_kmh.clear()

    def get_speed(self, track_id):
        """Vận tốc làm mượt (km/h) cho track_id."""
        return float(self._speed_ema_kmh.get(track_id, 0.0))

    def is_speeding(self, track_id, speed_limit_kmh=60):
        """Kiểm tra có đang vượt quá tốc độ không"""
        speed = self.get_speed(track_id)
        return speed > speed_limit_kmh