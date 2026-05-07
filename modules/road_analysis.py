"""
Module phân tích mặt đường - Ổn định hóa với Morphological & Nhớ khung hình
"""

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

class RoadAnalyzer:
    """
    Phân tích chất lượng lòng đường (Bản Chống rung/nháy hình)
    Ứng dụng Morphology và Bộ đệm ký ức thời gian.
    """

    def __init__(self):
        self.defect_history = {} # Lưu lại ổ gà/vật thể theo toạ độ
        self.next_defect_id = 0

        # Chỉ vẽ dị vật khi tồn tại được N frame (chống nhấp nháy nhiễu mây)
        self.min_hits = 3
        # Số frame tối đa chịu đựng việc mất dạng trước khi xóa khỏi bộ nhớ
        self.max_coast = 3

        self.obstacle_count = 0

        logger.info("RoadAnalyzer initialized (Stabilized Version)")

    def analyze(self, road_region, vehicle_mask=None):
        if road_region is None or road_region.size == 0:
            return self._get_empty_result()

        # 1. Chuyển sang ảnh xám và khử nhiễu (Làm mịn gợn nhỏ mặt nhựa)
        gray = cv2.cvtColor(road_region, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        h, w = gray.shape

        # Khởi tạo mask cản xe đi qua (Che đi chỗ xe đi để ko nhìn bóng xe)
        if vehicle_mask is not None and vehicle_mask.size > 0:
            if vehicle_mask.shape[:2] != (h, w):
                vehicle_mask = cv2.resize(vehicle_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            # Thổi bự vehicle mask lên để loại trừ sạch bóng đè (dilation)
            dilated_vehicle_mask = cv2.dilate(vehicle_mask, np.ones((15, 15), np.uint8), iterations=2)
            analysis_mask = cv2.bitwise_not(dilated_vehicle_mask)
        else:
            analysis_mask = np.ones((h, w), dtype=np.uint8) * 255

        # --- 2. THUẬT TOÁN ĐẠI HÌNH THÁI HỌC (Morphology) THAY CHO CANNY ----
        kernel_shape = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

        # BlackHat: Tìm tất cả khu vực Rất Đen trên một nền nhám Sáng Xám (chính là Ổ gà)
        blackhat = cv2.morphologyEx(gray_blurred, cv2.MORPH_BLACKHAT, kernel_shape)
        # TopHat: Tìm khu vực nổi bật, Sáng Lên, Cộm hẳn lên so với mặt nhựa đường (Chướng ngại vật, cục gạch, rác bọc)
        tophat = cv2.morphologyEx(gray_blurred, cv2.MORPH_TOPHAT, kernel_shape)

        _, thresh_dark = cv2.threshold(blackhat, 40, 255, cv2.THRESH_BINARY)  # Lấy chốt vết xước/lõm sâu
        _, thresh_bright = cv2.threshold(tophat, 45, 255, cv2.THRESH_BINARY)  # Lấy chốt cộm nhô dị vật sáng

        # 3. Khoá vùng không xét trên vệt dính của xe tải đỗ ngang
        defect_dark_mask = cv2.bitwise_and(thresh_dark, thresh_dark, mask=analysis_mask)
        defect_bright_mask = cv2.bitwise_and(thresh_bright, thresh_bright, mask=analysis_mask)

        # Lấy thông tin ứng cử viên của Frame
        raw_candidates = []

        # Tiết chiết Vùng Rỗ/Tối Đen -> Ổ Gà / Vết nứt
        contours_dark, _ = cv2.findContours(defect_dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours_dark:
            area = cv2.contourArea(cnt)
            if area > 180: # Filter sàn lọt rác 180 pixel
                x, y, bw, bh = cv2.boundingRect(cnt)
                aspect_ratio = bw / float(bh)
                # Dạng hẹp vuốt -> Vết nứt (crack), khối thụp lại là Ổ gà (pothole)
                c_type = 'crack' if (aspect_ratio > 3.0 or aspect_ratio < 0.3) else 'pothole'
                raw_candidates.append({'rect': (x,y,bw,bh), 'type': c_type, 'contour': cnt})

        # Tiết chiết Cộm Sáng/Bao Nilong -> Chướng Ngại Vật
        contours_bright, _ = cv2.findContours(defect_bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours_bright:
            if cv2.contourArea(cnt) > 250: # Khối nhô phải có tí kích cỡ to hơn ruồi
                x, y, bw, bh = cv2.boundingRect(cnt)
                raw_candidates.append({'rect': (x,y,bw,bh), 'type': 'obstacle', 'contour': cnt})

        # --- 4. THEO DÕI VÀ LÀM LÌ KẾT QUẢ ĐẦU RA BẰNG VÙNG NHỚ TĨNH ----
        stable_defects = self._stabilize_defects(raw_candidates)

        pothole_contours = []
        crack_contours = []
        obstacle_contours = []
        water_contours = [] # Tắt tính rỗng water (Không uy tín ngoài ngày mưa lớn, giữ form mã trả List rỗng bảo toàn Giao diện k lỗi lỗi)

        # Đẩy kết quả trả Giao DIện
        for defect in stable_defects:
            ctype = defect['type']
            cnt = defect['contour']
            if ctype == 'pothole':
                pothole_contours.append(cnt)
            elif ctype == 'crack':
                crack_contours.append(cnt)
            elif ctype == 'obstacle':
                obstacle_contours.append(cnt)

                # Cập nhập biến cố Tổng quan Thống Lỗi Toàn Bộ
                pos_key = f"{defect['rect'][0] // 50}_{defect['rect'][1] // 50}"
                if not hasattr(self, 'tracked_obst'): self.tracked_obst = set()
                if pos_key not in self.tracked_obst:
                    self.tracked_obst.add(pos_key)
                    self.obstacle_count += 1

        p_detect = len(pothole_contours) > 0
        c_detect = len(crack_contours) > 0
        o_detect = len(obstacle_contours) > 0

        condition_parts = []
        if p_detect: condition_parts.append(f"{len(pothole_contours)} Ổ gà")
        if o_detect: condition_parts.append(f"{len(obstacle_contours)} Vật thể chặn")

        quality_score = max(0, 100 - len(pothole_contours)*12 - len(crack_contours)*4 - len(obstacle_contours)*15)

        if condition_parts:
            condition = f"⚠️ BẤT THƯỜNG: " + ", ".join(condition_parts)
        elif quality_score >= 80: condition = "✅ MẶT ĐƯỜNG RẤT TỐT"
        else: condition = "⚪ MẶT ĐƯỜNG TRUNG BÌNH/NỨT NHẸ"

        result = {
            'quality_score': quality_score,
            'condition': condition,
            'pothole_detected': p_detect,
            'crack_detected': c_detect,
            'water_detected': False,
            'obstacle_detected': o_detect,
            'obstacle_count': len(obstacle_contours),
            'total_obstacles': self.obstacle_count,
            'edge_density': 0.0,
            'dark_ratio': 0.0,
            'texture_score': 0.0,
            'pothole_count': len(pothole_contours),
            'crack_count': len(crack_contours),
            'water_count': 0,
            'pothole_contours': pothole_contours,
            'crack_contours': crack_contours,
            'water_contours': [],
            'obstacle_contours': obstacle_contours,
            'road_area': road_region.shape[0] * road_region.shape[1],
            'analysis_mask': analysis_mask
        }
        return result

    def _stabilize_defects(self, candidates):
        """Giữ vị trí ổn định cứng đét cho các Khối Dị tật tĩnh vật."""
        for state in self.defect_history.values():
            state['coast_frames'] += 1

        for cand in candidates:
            c_x, c_y, c_w, c_h = cand['rect']
            cx_cand = c_x + c_w/2
            cy_cand = c_y + c_h/2

            matched_id = None
            min_dist = 50.0  # 50 pixel radius threshold

            for defect_id, state in self.defect_history.items():
                s_x, s_y, s_w, s_h = state['rect']
                cx_state = s_x + s_w/2
                cy_state = s_y + s_h/2
                dist = np.hypot(cx_cand - cx_state, cy_cand - cy_state)

                # Check nếu trùng tâm toạ độ và Chung Đặc chẩn chủng loài loại hình
                if dist < min_dist and state['type'] == cand['type']:
                    matched_id = defect_id
                    min_dist = dist

            if matched_id is not None:
                # Tìm trúng ID cũ. Giảm frame quên đi, nới Hit cứng lên. Merge vịt Box mịn đúp !
                self.defect_history[matched_id]['coast_frames'] = 0
                self.defect_history[matched_id]['hits'] += 1

                old_rect = self.defect_history[matched_id]['rect']
                alpha = 0.5 # EMA Box tĩnh smoothing
                new_rect = (
                    int(old_rect[0]*alpha + c_x*(1-alpha)),
                    int(old_rect[1]*alpha + c_y*(1-alpha)),
                    int(old_rect[2]*alpha + c_w*(1-alpha)),
                    int(old_rect[3]*alpha + c_h*(1-alpha)),
                )
                self.defect_history[matched_id]['rect'] = new_rect
                self.defect_history[matched_id]['contour'] = cand['contour'] # Cập viền bo chuẩn ảnh moi update
            else:
                self.defect_history[self.next_defect_id] = {
                    'type': cand['type'],
                    'rect': cand['rect'],
                    'contour': cand['contour'],
                    'hits': 1,
                    'coast_frames': 0
                }
                self.next_defect_id += 1

        # Xoá vùng đệm bị Xe dừng/Xe Cản đứt Bóng Tối Trí Đánh nhớ xoá luôn ổ đi rỗng List!
        to_del = [uid for uid, state in self.defect_history.items() if state['coast_frames'] > self.max_coast]
        for uid in to_del: del self.defect_history[uid]

        # Cuối -> MÓC Trích Các Vật đạt múc Sắt định chuẩn uy tín !
        final_draws = []
        for uid, state in self.defect_history.items():
            if state['hits'] >= self.min_hits: # Dư thời 3 frame nãy mới báo tin ! Ko ảo lòi lồi kẹo nưa !
                 final_draws.append(state)

        return final_draws

    def reset_obstacle_count(self):
        """Reset bộ đếm ổ gà/vật liệu xá trễ từ cữ Load File/Mở Cam Mới """
        self.defect_history = {}
        self.next_defect_id = 0
        self.obstacle_count = 0
        self.tracked_obst = set()

    def _get_empty_result(self):
        return {
            'quality_score': 0, 'condition': 'KHÔNG CÓ MẶT ĐƯỜNG', 'pothole_detected': False,
            'crack_detected': False, 'water_detected': False, 'obstacle_detected': False,
            'obstacle_count': 0, 'total_obstacles': 0, 'edge_density': 0.0, 'dark_ratio': 0.0,
            'texture_score': 0.0, 'pothole_count': 0, 'crack_count': 0, 'water_count': 0,
            'pothole_contours': [], 'crack_contours': [], 'water_contours': [], 'obstacle_contours': [],
            'road_area': 0, 'analysis_mask': None
        }