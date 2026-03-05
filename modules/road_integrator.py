"""
Module tích hợp phân tích mặt đường
Kết hợp segmentation và analysis
"""

import cv2
import numpy as np
import logging
from .road_segmentation import RoadSegmentation
from .road_analysis import RoadAnalyzer

logger = logging.getLogger(__name__)

class RoadIntegrator:
    """
    Tích hợp quá trình phân tích mặt đường
    B1: Segmentation - tách lòng đường và vỉa hè
    B2: Analysis - phân tích chất lượng
    B3: Visualization - vẽ kết quả
    """

    def __init__(self):
        self.segmentation = RoadSegmentation()
        self.analyzer = RoadAnalyzer()
        logger.info("RoadIntegrator initialized")

    def process(self, frame):
        """
        Xử lý hoàn chỉnh: tách mặt đường + phân tích + vẽ

        Args:
            frame: Ảnh đầu vào

        Returns:
            result: Kết quả phân tích
            annotated_frame: Ảnh đã vẽ
        """
        h, w = frame.shape[:2]

        # BƯỚC 1: TÁCH LÒNG ĐƯỜNG VÀ VỈA HÈ
        road_mask, sidewalk_mask, roi_coords, seg_info = self.segmentation.extract_road_and_sidewalk(frame)

        # BƯỚC 2: LẤY VÙNG LÒNG ĐƯỜNG
        y1, y2, x1, x2 = roi_coords
        road_region = self.segmentation.get_road_region(frame, road_mask, roi_coords)

        # BƯỚC 3: PHÂN TÍCH LÒNG ĐƯỜNG
        analysis_result = self.analyzer.analyze(road_region)

        # BƯỚC 4: KẾT HỢP KẾT QUẢ
        result = {
            **analysis_result,
            'roi_coords': roi_coords,
            'road_mask': road_mask,
            'sidewalk_mask': sidewalk_mask,
            'road_area': seg_info.get('road_area', 0)
        }

        # BƯỚC 5: VẼ KẾT QUẢ
        annotated = self._draw_results(frame, result)

        return result, annotated

    def _draw_results(self, frame, result):
        """
        Vẽ kết quả lên frame
        """
        h, w = frame.shape[:2]
        y1, y2, x1, x2 = result['roi_coords']

        annotated = frame.copy()

        # ----- 1. TÔ MÀU VỈA HÈ (màu xám nhạt) -----
        if result['sidewalk_mask'] is not None and result['sidewalk_mask'].size > 0:
            sidewalk_overlay = annotated.copy()
            sidewalk_overlay[result['sidewalk_mask'] > 0] = (128, 128, 128)  # Màu xám
            cv2.addWeighted(sidewalk_overlay, 0.3, annotated, 0.7, 0, annotated)
            cv2.putText(annotated, "VIA HE", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)

        # ----- 2. TÔ MÀU LÒNG ĐƯỜNG CHÍNH (màu xanh nhạt) -----
        if result['road_mask'] is not None and result['road_mask'].size > 0:
            road_overlay = annotated.copy()
            road_overlay[result['road_mask'] > 0] = (0, 255, 0)  # Màu xanh lá
            cv2.addWeighted(road_overlay, 0.1, annotated, 0.9, 0, annotated)

        # ----- 3. VẼ VIỀN LÒNG ĐƯỜNG CHÍNH -----
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(annotated, "LONG DUONG CHINH",
                   (x1+10, y1+30), cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, (0, 255, 0), 2)

        # ----- 4. VẼ Ổ GÀ (màu đỏ) -----
        for contour in result['pothole_contours']:
            # Điều chỉnh contour về tọa độ gốc
            contour_abs = contour.copy()
            contour_abs[:, :, 0] += x1
            contour_abs[:, :, 1] += y1

            cv2.drawContours(annotated, [contour_abs], -1, (0, 0, 255), 2)

            # Ghi nhãn
            M = cv2.moments(contour_abs)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.putText(annotated, "O GA", (cx-30, cy-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # ----- 5. VẼ VẾT NỨT (màu cam) -----
        for contour in result['crack_contours']:
            contour_abs = contour.copy()
            contour_abs[:, :, 0] += x1
            contour_abs[:, :, 1] += y1
            cv2.drawContours(annotated, [contour_abs], -1, (0, 165, 255), 2)

        # ----- 6. BẢNG THÔNG TIN -----
        info_x, info_y = w - 450, 30
        cv2.rectangle(annotated, (info_x-10, info_y-30),
                     (info_x+440, info_y+160), (0, 0, 0), -1)

        # Tiêu đề
        cv2.putText(annotated, "PHAN TICH MAT DUONG",
                   (info_x, info_y-5), cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, (255, 255, 255), 2)

        # Thông tin
        y_offset = info_y + 20
        lines = [
            f"Tinh trang: {result['condition']}",
            f"Chat luong: {result['quality_score']}%",
            f"O ga: {result['pothole_count']} | Vet nut: {result['crack_count']}",
            f"Mat do canh: {result['edge_density']:.3f}",
            f"Vung phan tich: ({x1},{y1})-({x2},{y2})",
            f"Dien tich duong: {result['road_area']} px"
        ]

        for i, line in enumerate(lines):
            color = (0, 0, 255) if i == 1 and result['pothole_detected'] else (255, 255, 255)
            cv2.putText(annotated, line, (info_x, y_offset + i*20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return annotated