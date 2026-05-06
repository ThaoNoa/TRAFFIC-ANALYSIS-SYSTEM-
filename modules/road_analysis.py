"""
Module phân tích mặt đường
Nhiệm vụ: Phân tích chất lượng lòng đường chính
Phân biệt: ổ gà, vết nứt (tĩnh) vs phương tiện (động)
"""

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

class RoadAnalyzer:
    """
    Phân tích chất lượng lòng đường
    Chỉ làm việc trên vùng lòng đường đã được tách
    Phân biệt rõ: ổ gà/vết nứt là tĩnh, phương tiện là động
    """

    def __init__(self):
        # Ngưỡng phát hiện
        self.pothole_threshold = 0.08
        self.crack_threshold = 0.12
        self.water_threshold = 0.1
        self.obstacle_threshold = 0.05  # Ngưỡng cho chướng ngại vật

        # Kernel cho morphology
        self.kernel_small = np.ones((3, 3), np.uint8)
        self.kernel_medium = np.ones((5, 5), np.uint8)

        # Đếm chướng ngại vật
        self.obstacle_count = 0
        self.obstacle_history = {}  # Lưu vị trí chướng ngại vật để tránh đếm trùng

        logger.info("RoadAnalyzer initialized")

    def analyze(self, road_region, vehicle_mask=None):
        """
        Phân tích chất lượng lòng đường, phân biệt tĩnh/động

        Args:
            road_region: Ảnh chỉ chứa lòng đường (đã crop)
            vehicle_mask: Mask các phương tiện (để loại trừ khỏi phân tích hư hỏng)

        Returns:
            dict: Kết quả phân tích
        """
        if road_region is None or road_region.size == 0:
            return self._get_empty_result()

        # Chuyển sang grayscale
        gray = cv2.cvtColor(road_region, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Nếu có vehicle_mask, tạo mask chỉ phân tích vùng không có xe
        if vehicle_mask is not None and vehicle_mask.size > 0:
            if vehicle_mask.shape[:2] != (h, w):
                vehicle_mask = cv2.resize(vehicle_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            analysis_mask = cv2.bitwise_not(vehicle_mask)
        else:
            analysis_mask = np.ones((h, w), dtype=np.uint8) * 255

        # ----- 1. PHÁT HIỆN Ổ GÀ -----
        edges = cv2.Canny(gray, 30, 100)
        dark_regions = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)[1]

        edges_filtered = cv2.bitwise_and(edges, edges, mask=analysis_mask)
        dark_regions_filtered = cv2.bitwise_and(dark_regions, dark_regions, mask=analysis_mask)

        contours, _ = cv2.findContours(edges_filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        pothole_contours = []
        crack_contours = []
        water_contours = []
        obstacle_contours = []  # THÊM: danh sách chướng ngại vật

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue

            x, y, w_box, h_box = cv2.boundingRect(contour)
            aspect_ratio = w_box / h_box if h_box > 0 else 0
            circularity = 4 * np.pi * area / (perimeter * perimeter)

            roi_dark = dark_regions_filtered[y:y+h_box, x:x+w_box]
            dark_ratio_in_contour = np.sum(roi_dark > 0) / roi_dark.size if roi_dark.size > 0 else 0

            # === THÊM: PHÁT HIỆN CHƯỚNG NGẠI VẬT ===
            # Chướng ngại vật = vật thể lạ trên đường (không phải ổ gà, không phải vết nứt)
            is_obstacle = False

            # Phân loại
            if 150 < area < 5000:
                if aspect_ratio > 3 or aspect_ratio < 0.33:
                    if dark_ratio_in_contour > 0.3:
                        crack_contours.append(contour)
                elif circularity < 0.7:
                    if dark_ratio_in_contour > 0.4:
                        pothole_contours.append(contour)
                    elif dark_ratio_in_contour < 0.2:
                        roi = gray[y:y+h_box, x:x+w_box]
                        if np.std(roi) < 30:
                            water_contours.append(contour)
                        else:
                            # Vật thể lạ không phải ổ gà, nước, vết nứt
                            is_obstacle = True
                            obstacle_contours.append(contour)
                else:
                    # Contour tròn nhưng không phải ổ gà -> có thể là chướng ngại vật
                    if dark_ratio_in_contour < 0.3 and area > 300:
                        is_obstacle = True
                        obstacle_contours.append(contour)

        # Cập nhật số lượng chướng ngại vật (tránh đếm trùng)
        self._update_obstacle_count(obstacle_contours, gray.shape)

        # ----- 2. PHÂN TÍCH TEXTURE -----
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        laplacian = cv2.filter2D(gray, cv2.CV_64F, kernel)
        masked_laplacian = cv2.bitwise_and(laplacian.astype(np.uint8),
                                           laplacian.astype(np.uint8),
                                           mask=analysis_mask)
        if np.sum(analysis_mask > 0) > 0:
            texture_score = np.std(masked_laplacian[masked_laplacian > 0])
        else:
            texture_score = 0

        # ----- 3. TÍNH ĐIỂM CHẤT LƯỢNG -----
        edge_density = np.sum(edges_filtered > 0) / np.sum(analysis_mask > 0) if np.sum(analysis_mask > 0) > 0 else 0
        dark_ratio = np.sum(dark_regions_filtered > 0) / np.sum(analysis_mask > 0) if np.sum(analysis_mask > 0) > 0 else 0

        quality_score = self._calculate_quality_score(
            edge_density, dark_ratio, texture_score,
            len(pothole_contours), len(crack_contours), len(water_contours)
        )

        # ----- 4. XÁC ĐỊNH TÌNH TRẠNG -----
        pothole_detected = len(pothole_contours) > 0
        crack_detected = len(crack_contours) > 0
        water_detected = len(water_contours) > 0
        obstacle_detected = len(obstacle_contours) > 0

        condition_parts = []
        if pothole_detected:
            condition_parts.append(f"{len(pothole_contours)} ổ gà")
        if crack_detected:
            condition_parts.append(f"{len(crack_contours)} vết nứt")
        if water_detected:
            condition_parts.append(f"{len(water_contours)} vũng nước")
        if obstacle_detected:
            condition_parts.append(f"{len(obstacle_contours)} chướng ngại vật")

        if condition_parts:
            condition = f"⚠️ PHÁT HIỆN: " + ", ".join(condition_parts)
        elif quality_score >= 80:
            condition = "✅ MẶT ĐƯỜNG TỐT"
        elif quality_score >= 60:
            condition = "⚪ MẶT ĐƯỜNG TRUNG BÌNH"
        elif quality_score >= 40:
            condition = "🟠 MẶT ĐƯỜNG XẤU"
        else:
            condition = "🔴 MẶT ĐƯỜNG RẤT XẤU"

        result = {
            'quality_score': quality_score,
            'condition': condition,
            'pothole_detected': pothole_detected,
            'crack_detected': crack_detected,
            'water_detected': water_detected,
            'obstacle_detected': obstacle_detected,  # THÊM
            'obstacle_count': len(obstacle_contours),  # THÊM
            'total_obstacles': self.obstacle_count,  # THÊM: tổng số chướng ngại vật đã phát hiện
            'edge_density': float(edge_density),
            'dark_ratio': float(dark_ratio),
            'texture_score': float(texture_score),
            'pothole_count': len(pothole_contours),
            'crack_count': len(crack_contours),
            'water_count': len(water_contours),
            'pothole_contours': pothole_contours,
            'crack_contours': crack_contours,
            'water_contours': water_contours,
            'obstacle_contours': obstacle_contours,  # THÊM
            'road_area': road_region.shape[0] * road_region.shape[1],
            'analysis_mask': analysis_mask
        }

        return result

    def _update_obstacle_count(self, obstacle_contours, frame_shape):
        """Cập nhật số lượng chướng ngại vật, tránh đếm trùng"""
        h, w = frame_shape[:2] if isinstance(frame_shape, tuple) else frame_shape

        for contour in obstacle_contours:
            M = cv2.moments(contour)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])

                # Tạo key duy nhất cho vị trí (chia lưới 50x50 pixel)
                grid_x = cx // 50
                grid_y = cy // 50
                key = f"{grid_x}_{grid_y}"

                # Nếu chưa có trong lịch sử, thêm vào và tăng count
                if key not in self.obstacle_history:
                    self.obstacle_history[key] = {
                        'count': 1,
                        'first_seen': len(self.obstacle_history),
                        'position': (cx, cy)
                    }
                    self.obstacle_count += 1

    def reset_obstacle_count(self):
        """Reset bộ đếm chướng ngại vật (gọi khi bắt đầu video/camera mới)"""
        self.obstacle_count = 0
        self.obstacle_history = {}

    def _calculate_quality_score(self, edge_density, dark_ratio, texture,
                                  pothole_count, crack_count, water_count):
        """Tính điểm chất lượng đường"""
        score = 100
        score -= min(edge_density * 150, 25)
        score -= min(dark_ratio * 250, 40)
        score -= min(texture / 4.0, 15)
        score -= min(pothole_count * 8, 30)
        score -= min(crack_count * 5, 15)
        score -= min(water_count * 4, 10)
        return int(max(0, min(100, score)))

    def _get_empty_result(self):
        """Kết quả mặc định khi không có vùng đường"""
        return {
            'quality_score': 0,
            'condition': 'KHÔNG CÓ MẶT ĐƯỜNG',
            'pothole_detected': False,
            'crack_detected': False,
            'water_detected': False,
            'obstacle_detected': False,
            'obstacle_count': 0,
            'total_obstacles': self.obstacle_count,
            'edge_density': 0.0,
            'dark_ratio': 0.0,
            'texture_score': 0.0,
            'pothole_count': 0,
            'crack_count': 0,
            'water_count': 0,
            'pothole_contours': [],
            'crack_contours': [],
            'water_contours': [],
            'obstacle_contours': [],
            'road_area': 0,
            'analysis_mask': None
        }