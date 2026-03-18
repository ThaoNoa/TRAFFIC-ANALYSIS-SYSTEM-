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
        self.water_threshold = 0.1  # Ngưỡng phát hiện vũng nước

        # Kernel cho morphology
        self.kernel_small = np.ones((3, 3), np.uint8)
        self.kernel_medium = np.ones((5, 5), np.uint8)

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
            # Resize vehicle_mask về cùng kích thước với road_region nếu cần
            if vehicle_mask.shape[:2] != (h, w):
                vehicle_mask = cv2.resize(vehicle_mask, (w, h), interpolation=cv2.INTER_NEAREST)

            # Mask phân tích: chỉ phân tích vùng không có xe
            analysis_mask = cv2.bitwise_not(vehicle_mask)
        else:
            analysis_mask = np.ones((h, w), dtype=np.uint8) * 255

        # ----- 1. PHÁT HIỆN Ổ GÀ (dựa trên cạnh và vùng tối) -----
        edges = cv2.Canny(gray, 30, 100)
        dark_regions = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)[1]

        # Chỉ phân tích trên vùng không có xe
        edges_filtered = cv2.bitwise_and(edges, edges, mask=analysis_mask)
        dark_regions_filtered = cv2.bitwise_and(dark_regions, dark_regions, mask=analysis_mask)

        # Tìm contours trên edges để phát hiện ổ gà
        contours, _ = cv2.findContours(edges_filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        pothole_contours = []
        crack_contours = []
        water_contours = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:  # Bỏ qua các contour quá nhỏ
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue

            # Tính các đặc trưng hình học
            x, y, w_box, h_box = cv2.boundingRect(contour)
            aspect_ratio = w_box / h_box if h_box > 0 else 0
            circularity = 4 * np.pi * area / (perimeter * perimeter)

            # Tính độ sâu (dựa trên độ tối của vùng)
            roi_dark = dark_regions_filtered[y:y+h_box, x:x+w_box]
            dark_ratio_in_contour = np.sum(roi_dark > 0) / roi_dark.size if roi_dark.size > 0 else 0

            # Phân loại dựa trên hình dạng và đặc điểm
            if 150 < area < 5000:
                if aspect_ratio > 3 or aspect_ratio < 0.33:  # Dạng kéo dài
                    if dark_ratio_in_contour > 0.3:  # Vết nứt thường tối
                        crack_contours.append(contour)
                elif circularity < 0.7:  # Dạng không đều
                    if dark_ratio_in_contour > 0.4:  # Ổ gà thường tối
                        pothole_contours.append(contour)
                    elif dark_ratio_in_contour < 0.2:  # Vũng nước thường sáng bóng
                        # Kiểm tra độ phẳng để phân biệt vũng nước
                        roi = gray[y:y+h_box, x:x+w_box]
                        if np.std(roi) < 30:  # Vũng nước có texture đồng nhất
                            water_contours.append(contour)

        # ----- 2. PHÂN TÍCH TEXTURE TỔNG THỂ -----
        kernel = np.array([[-1, -1, -1],
                          [-1,  8, -1],
                          [-1, -1, -1]])
        laplacian = cv2.filter2D(gray, cv2.CV_64F, kernel)
        texture_score = np.std(laplacian)

        # Chỉ tính trên vùng không có xe
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

        condition_parts = []
        if pothole_detected:
            condition_parts.append(f"{len(pothole_contours)} ổ gà")
        if crack_detected:
            condition_parts.append(f"{len(crack_contours)} vết nứt")
        if water_detected:
            condition_parts.append(f"{len(water_contours)} vũng nước")

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
            'edge_density': float(edge_density),
            'dark_ratio': float(dark_ratio),
            'texture_score': float(texture_score),
            'pothole_count': len(pothole_contours),
            'crack_count': len(crack_contours),
            'water_count': len(water_contours),
            'pothole_contours': pothole_contours,
            'crack_contours': crack_contours,
            'water_contours': water_contours,
            'road_area': road_region.shape[0] * road_region.shape[1],
            'analysis_mask': analysis_mask  # Trả về để vẽ
        }

        return result

    def _calculate_quality_score(self, edge_density, dark_ratio, texture,
                                  pothole_count, crack_count, water_count):
        """Tính điểm chất lượng đường"""
        score = 100

        # Mật độ cạnh cao -> đường gồ ghề
        score -= min(edge_density * 150, 25)

        # Nhiều vùng tối -> có ổ gà
        score -= min(dark_ratio * 250, 40)

        # Texture phức tạp -> bề mặt không đều
        score -= min(texture / 4.0, 15)

        # Có ổ gà thực tế
        score -= min(pothole_count * 8, 30)

        # Có vết nứt
        score -= min(crack_count * 5, 15)

        # Có vũng nước
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
            'edge_density': 0.0,
            'dark_ratio': 0.0,
            'texture_score': 0.0,
            'pothole_count': 0,
            'crack_count': 0,
            'water_count': 0,
            'pothole_contours': [],
            'crack_contours': [],
            'water_contours': [],
            'road_area': 0,
            'analysis_mask': None
        }