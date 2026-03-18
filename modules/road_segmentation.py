"""
Module phân đoạn mặt đường
Nhiệm vụ: Phân tách lòng đường chính và vỉa hè
"""

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

class RoadSegmentation:
    """
    Phân đoạn ảnh để tách lòng đường chính và vỉa hè
    Giả định camera chĩa xuống, chỉ thấy mặt đường
    """

    def __init__(self):
        # Ngưỡng màu cho các đối tượng
        # Lòng đường (màu xám đậm, asphalt)
        self.road_hsv_lower = np.array([0, 0, 30])
        self.road_hsv_upper = np.array([180, 50, 180])

        # Vỉa hè (màu sáng hơn, thường có màu xám nhạt hoặc màu gạch)
        self.sidewalk_hsv_lower = np.array([0, 0, 150])
        self.sidewalk_hsv_upper = np.array([180, 60, 255])

        # Vạch kẻ đường (màu trắng, vàng)
        self.lane_line_lower = np.array([0, 0, 200])
        self.lane_line_upper = np.array([180, 50, 255])

        # Kernel cho morphology
        self.kernel_small = np.ones((3, 3), np.uint8)
        self.kernel_medium = np.ones((5, 5), np.uint8)
        self.kernel_large = np.ones((15, 15), np.uint8)

        logger.info("RoadSegmentation initialized (camera pointing down)")

    def extract_road_and_sidewalk(self, frame):
        """
        Tách lòng đường chính và vỉa hè từ ảnh

        Args:
            frame: Ảnh BGR đầu vào

        Returns:
            road_mask: Mask lòng đường chính
            sidewalk_mask: Mask vỉa hè
            roi_coords: Tọa độ [y1, y2, x1, x2] của vùng quan tâm
            debug_info: Dict chứa thông tin debug
        """
        h, w = frame.shape[:2]
        debug_info = {}

        # Bước 1: Chuyển sang HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Bước 2: Tạo mask cho lòng đường và vỉa hè dựa trên màu sắc
        road_color_mask = cv2.inRange(hsv, self.road_hsv_lower, self.road_hsv_upper)
        sidewalk_color_mask = cv2.inRange(hsv, self.sidewalk_hsv_lower, self.sidewalk_hsv_upper)
        lane_line_mask = cv2.inRange(hsv, self.lane_line_lower, self.lane_line_upper)

        # Bước 3: Phân tích texture
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Texture của lòng đường thường đồng nhất hơn vỉa hè
        # Tính độ lệch chuẩn cục bộ
        block_size = 20
        texture_std = np.zeros((h, w), dtype=np.float32)

        for y in range(0, h, block_size):
            for x in range(0, w, block_size):
                y_end = min(y + block_size, h)
                x_end = min(x + block_size, w)
                block = gray[y:y_end, x:x_end]
                texture_std[y:y_end, x:x_end] = np.std(block)

        # Chuẩn hóa
        texture_std = cv2.normalize(texture_std, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Vùng texture thấp (đồng nhất) có khả năng là lòng đường
        low_texture_mask = cv2.threshold(texture_std, 50, 255, cv2.THRESH_BINARY_INV)[1]

        # Bước 4: Kết hợp các thông tin
        # Lòng đường: màu phù hợp + texture thấp
        road_mask = cv2.bitwise_and(road_color_mask, low_texture_mask)

        # Vỉa hè: màu vỉa hè hoặc texture cao + không phải lòng đường
        sidewalk_mask = cv2.bitwise_or(sidewalk_color_mask,
                                       cv2.bitwise_and(cv2.bitwise_not(low_texture_mask),
                                                      cv2.bitwise_not(road_color_mask)))

        # Thêm vạch kẻ đường vào lòng đường
        road_mask = cv2.bitwise_or(road_mask, lane_line_mask)

        # Bước 5: Làm sạch bằng morphology
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, self.kernel_medium)
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, self.kernel_small)

        sidewalk_mask = cv2.morphologyEx(sidewalk_mask, cv2.MORPH_CLOSE, self.kernel_medium)
        sidewalk_mask = cv2.morphologyEx(sidewalk_mask, cv2.MORPH_OPEN, self.kernel_small)

        # Bước 6: Tìm contour lớn nhất cho lòng đường
        contours, _ = cv2.findContours(road_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Lấy contour lớn nhất
            main_road = max(contours, key=cv2.contourArea)

            # Tạo mask mới chỉ chứa contour lớn nhất
            final_road_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(final_road_mask, [main_road], -1, 255, -1)

            # Lấy bounding box
            x, y, w_road, h_road = cv2.boundingRect(main_road)
            roi_coords = [y, y+h_road, x, x+w_road]

            debug_info['road_area'] = cv2.contourArea(main_road)
            debug_info['method'] = 'contour_based'

            # Lọc sidewalk mask chỉ lấy các vùng không phải road
            sidewalk_mask = cv2.bitwise_and(sidewalk_mask, cv2.bitwise_not(final_road_mask))

            return final_road_mask, sidewalk_mask, roi_coords, debug_info

        # Fallback: nếu không tìm thấy contour, lấy phần trung tâm
        center_x, center_y = w // 2, h // 2
        roi_width, roi_height = w // 2, h // 2
        x1 = max(0, center_x - roi_width // 2)
        y1 = max(0, center_y - roi_height // 2)
        x2 = min(w, center_x + roi_width // 2)
        y2 = min(h, center_y + roi_height // 2)

        final_road_mask = np.zeros((h, w), dtype=np.uint8)
        final_road_mask[y1:y2, x1:x2] = 255
        roi_coords = [y1, y2, x1, x2]

        debug_info['method'] = 'fallback'
        debug_info['road_area'] = (x2 - x1) * (y2 - y1)

        return final_road_mask, sidewalk_mask, roi_coords, debug_info

    def detect_lane_lines(self, frame):
        """
        Phát hiện vạch kẻ đường để xác định làn đường

        Args:
            frame: Ảnh BGR đầu vào

        Returns:
            line_mask: Mask các vạch kẻ đường
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Phát hiện đường thẳng bằng Hough Transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50,
                                 minLineLength=50, maxLineGap=30)

        h, w = frame.shape[:2]
        line_mask = np.zeros((h, w), dtype=np.uint8)

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]

                # Tính góc của đường thẳng
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

                # Giữ các đường gần thẳng đứng hoặc nằm ngang
                # (vạch kẻ đường thường dọc hoặc ngang)
                if abs(angle) < 20 or abs(angle - 90) < 20 or abs(angle - 180) < 20:
                    cv2.line(line_mask, (x1, y1), (x2, y2), 255, 3)

        # Mở rộng các đường thẳng
        line_mask = cv2.dilate(line_mask, self.kernel_small, iterations=2)

        return line_mask

    def get_road_region(self, frame, road_mask, roi_coords):
        """
        Lấy vùng lòng đường đã crop

        Args:
            frame: Ảnh gốc
            road_mask: Mask lòng đường
            roi_coords: Tọa độ [y1,y2,x1,x2]

        Returns:
            road_region: Ảnh chỉ chứa lòng đường (đã crop)
        """
        y1, y2, x1, x2 = roi_coords
        road_region = frame[y1:y2, x1:x2].copy()

        # Áp dụng mask để loại bỏ phần còn lại
        if road_mask is not None and road_mask.size > 0:
            mask_crop = road_mask[y1:y2, x1:x2]
            road_region = cv2.bitwise_and(road_region, road_region, mask=mask_crop)

        return road_region

    def validate_road_mask(self, road_mask, frame):
        """
        Xác thực mask lòng đường có hợp lệ không

        Args:
            road_mask: Mask lòng đường
            frame: Ảnh gốc

        Returns:
            bool: True nếu mask hợp lệ
        """
        if road_mask is None:
            return False

        h, w = frame.shape[:2]

        # 1. Kiểm tra diện tích
        road_area = np.sum(road_mask > 0)
        if road_area < w * h * 0.1:  # Quá nhỏ (<10% ảnh)
            return False

        # 2. Kiểm tra vị trí - lòng đường thường ở trung tâm
        y_indices = np.where(np.sum(road_mask, axis=1) > 0)[0]
        x_indices = np.where(np.sum(road_mask, axis=0) > 0)[0]

        if len(y_indices) == 0 or len(x_indices) == 0:
            return False

        # Kiểm tra độ phủ
        coverage = road_area / (w * h)
        if coverage < 0.15:  # Quá ít
            return False

        return True