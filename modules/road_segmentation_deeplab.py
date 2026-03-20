"""
Module phân đoạn mặt đường sử dụng DeepLabv3+ - Phiên bản ổn định, không phụ thuộc skimage
"""

import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
import cv2
import numpy as np
import logging
from typing import Tuple, Optional, List
from scipy import ndimage

logger = logging.getLogger(__name__)


class RoadSegmentationDeepLab:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu',
                 model_path: Optional[str] = None,
                 input_size: Tuple[int, int] = (512, 512),
                 use_ensemble: bool = True):
        """
        Khởi tạo module phân đoạn mặt đường

        Args:
            device: Thiết bị tính toán
            model_path: Đường dẫn đến model đã fine-tune (None nếu dùng pre-trained)
            input_size: Kích thước đầu vào cho model
            use_ensemble: Sử dụng ensemble với các phương pháp truyền thống
        """
        self.device = torch.device(device)
        self.input_size = input_size
        self.use_ensemble = use_ensemble

        # Khởi tạo model DeepLabv3+
        self.model = self._initialize_model(model_path)
        self.model.eval()

        # Transform cho ảnh đầu vào
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # Tham số cho xử lý hậu kỳ
        self.post_process_params = {
            'min_road_area': 500,  # Diện tích tối thiểu của mặt đường (pixel)
            'morphology_kernel': 5,  # Kernel cho morphological operations
            'temporal_smoothing': 0.7,  # Hệ số làm mượt theo thời gian
            'road_threshold': 0.5,  # Ngưỡng xác suất cho mặt đường
        }

        # Bộ nhớ tạm cho làm mượt theo thời gian
        self.prev_mask = None
        self.temporal_buffer = []
        self.buffer_size = 5

        logger.info(f"RoadSegmentationDeepLab initialized on {device}")

    def _initialize_model(self, model_path: Optional[str]) -> nn.Module:
        """Khởi tạo model DeepLabv3+"""
        # Sử dụng resnet101 làm backbone
        model = models.segmentation.deeplabv3_resnet101(pretrained=(model_path is None))

        # Nếu có model đã fine-tune, load weights
        if model_path is not None:
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded fine-tuned model from {model_path}")

        # Thay đổi classifier cho bài toán nhị phân (đường/không đường)
        if model_path is None:
            model.classifier = self._get_binary_classifier()

        return model.to(self.device)

    def _get_binary_classifier(self) -> nn.Module:
        """Tạo classifier cho bài toán nhị phân (2 classes)"""
        return nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 2, kernel_size=1)  # 2 classes: road và non-road
        )

    def segment_road(self, frame: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        """
        Phân đoạn mặt đường với các kỹ thuật nâng cao độ ổn định

        Args:
            frame: Ảnh BGR đầu vào

        Returns:
            road_mask: Mask nhị phân của mặt đường (255 là đường, 0 là không)
            roi_coords: Tọa độ vùng quan tâm [y1, y2, x1, x2]
        """
        original_h, original_w = frame.shape[:2]

        # 1. Deep learning prediction
        dl_mask = self._deep_learning_prediction(frame)

        # 2. Ensemble với phương pháp truyền thống nếu cần
        if self.use_ensemble:
            traditional_mask = self._traditional_segmentation(frame)
            dl_mask = self._ensemble_masks(dl_mask, traditional_mask)

        # 3. Xử lý hậu kỳ nâng cao
        road_mask = self._post_processing(dl_mask, original_h, original_w)

        # 4. Làm mượt theo thời gian
        road_mask = self._temporal_smoothing(road_mask)

        # 5. Xác định ROI
        roi_coords = self._get_roi_coords(road_mask, original_h, original_w)

        return road_mask, roi_coords

    def _deep_learning_prediction(self, frame: np.ndarray) -> np.ndarray:
        """Dự đoán bằng deep learning"""
        # Chuyển đổi ảnh đầu vào
        input_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_resized = cv2.resize(input_rgb, self.input_size)
        input_tensor = self.transform(input_resized).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)['out'][0]  # Shape: (2, H, W)

            # Softmax để có xác suất
            probabilities = torch.softmax(output, dim=0)
            road_prob = probabilities[1].cpu().numpy()  # Class 1 là mặt đường

        return road_prob

    def _traditional_segmentation(self, frame: np.ndarray) -> np.ndarray:
        """
        Phân đoạn mặt đường bằng phương pháp truyền thống dựa trên:
        - Màu sắc (HSV)
        - Vị trí không gian
        """
        h, w = frame.shape[:2]

        # Chuyển sang HSV để phân tích màu sắc
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Ngưỡng màu cho mặt đường (có thể điều chỉnh)
        # Màu xám, đen, xám đậm thường là đường
        lower_gray = np.array([0, 0, 40])
        upper_gray = np.array([180, 50, 180])

        # Màu của mặt đường bê tông/nhựa
        road_color_mask = cv2.inRange(hsv, lower_gray, upper_gray)

        # Lọc theo vị trí (đường thường ở phần dưới ảnh)
        bottom_region = np.zeros((h, w), dtype=np.uint8)
        bottom_region[h * 2 // 3:, :] = 255

        # Kết hợp
        traditional_mask = cv2.bitwise_and(road_color_mask, bottom_region)

        # Làm mịn mask
        kernel = np.ones((5, 5), np.uint8)
        traditional_mask = cv2.morphologyEx(traditional_mask, cv2.MORPH_CLOSE, kernel)
        traditional_mask = cv2.medianBlur(traditional_mask, 5)

        return traditional_mask / 255.0

    def _ensemble_masks(self, dl_mask: np.ndarray, trad_mask: np.ndarray,
                        alpha: float = 0.7) -> np.ndarray:
        """
        Kết hợp kết quả từ deep learning và phương pháp truyền thống
        """
        # Resize traditional mask về cùng kích thước với dl_mask
        trad_resized = cv2.resize(trad_mask, self.input_size)

        # Weighted ensemble
        ensemble_mask = alpha * dl_mask + (1 - alpha) * trad_resized

        return ensemble_mask

    def _post_processing(self, mask: np.ndarray, orig_h: int, orig_w: int) -> np.ndarray:
        """
        Xử lý hậu kỳ nâng cao để cải thiện chất lượng mask
        """
        # Resize về kích thước gốc
        mask_resized = cv2.resize(mask, (orig_w, orig_h))

        # Binary threshold
        threshold = self.post_process_params['road_threshold']
        binary_mask = (mask_resized > threshold).astype(np.uint8) * 255

        # 1. Morphological operations để loại bỏ nhiễu
        kernel_size = self.post_process_params['morphology_kernel']
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        # Closing để lấp đầy lỗ hổng
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        # Opening để loại bỏ nhiễu
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

        # 2. Loại bỏ các vùng nhỏ
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, 8)
        min_area = self.post_process_params['min_road_area']

        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            valid_labels = np.where(areas >= min_area)[0] + 1

            # Tạo mask mới chỉ với các vùng hợp lệ
            clean_mask = np.zeros_like(binary_mask)
            for label in valid_labels:
                clean_mask[labels == label] = 255

            binary_mask = clean_mask

        # 3. Xử lý biên để làm mượt
        binary_mask = cv2.medianBlur(binary_mask, 5)

        # 4. Xác định vùng đường chính (vùng lớn nhất nối với cạnh dưới)
        binary_mask = self._get_connected_to_bottom(binary_mask)

        return binary_mask

    def _get_connected_to_bottom(self, mask: np.ndarray) -> np.ndarray:
        """
        Lấy vùng đường kết nối với cạnh dưới của ảnh
        Giả sử mặt đường luôn nối với cạnh dưới (gần camera)
        """
        h, w = mask.shape

        # Tìm các pixel ở cạnh dưới
        bottom_pixels = []
        for x in range(w):
            if mask[h - 1, x] > 0:
                bottom_pixels.append((x, h - 1))

        if not bottom_pixels:
            # Nếu không có pixel nào ở cạnh dưới, giữ nguyên mask
            return mask

        # Label các vùng kết nối
        labeled_mask, num_labels = ndimage.label(mask)

        # Tìm label của các vùng kết nối với cạnh dưới
        valid_labels = set()
        for x, y in bottom_pixels:
            if y < h and x < w:
                label = labeled_mask[y, x]
                if label > 0:
                    valid_labels.add(label)

        # Giữ lại các vùng hợp lệ
        result = np.zeros_like(mask)
        for label in valid_labels:
            result[labeled_mask == label] = 255

        return result

    def _temporal_smoothing(self, current_mask: np.ndarray) -> np.ndarray:
        """Làm mượt mask theo thời gian để tăng tính ổn định"""
        if self.prev_mask is None:
            self.prev_mask = current_mask
            self.temporal_buffer = [current_mask] * self.buffer_size
            return current_mask

        # Lưu vào buffer
        self.temporal_buffer.pop(0)
        self.temporal_buffer.append(current_mask)

        # Trung bình có trọng số
        weights = np.exp(-np.linspace(0, 2, self.buffer_size))
        weights /= weights.sum()

        smoothed = np.zeros_like(current_mask, dtype=np.float32)
        for i, mask in enumerate(self.temporal_buffer):
            smoothed += weights[i] * mask.astype(np.float32)

        smoothed = (smoothed > 127).astype(np.uint8) * 255
        self.prev_mask = smoothed

        return smoothed

    def _get_roi_coords(self, mask: np.ndarray, h: int, w: int) -> List[int]:
        """
        Xác định vùng quan tâm dựa trên mask mặt đường
        """
        # Tìm contour lớn nhất
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Lấy contour lớn nhất
            main_contour = max(contours, key=cv2.contourArea)

            # Lấy bounding box
            x, y, w_road, h_road = cv2.boundingRect(main_contour)

            # Mở rộng bounding box một chút để có vùng an toàn
            margin = 20
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(w, x + w_road + margin)
            y2 = min(h, y + h_road + margin)

            roi_coords = [y1, y2, x1, x2]
        else:
            # Nếu không tìm thấy, sử dụng vùng mặc định (1/3 dưới ảnh)
            y_start = h * 2 // 3
            roi_coords = [y_start, h, 0, w]

        return roi_coords

    def update_post_process_params(self, **kwargs):
        """Cập nhật tham số xử lý hậu kỳ"""
        self.post_process_params.update(kwargs)
        logger.info(f"Updated post-process parameters: {kwargs}")

    def set_road_threshold(self, threshold: float):
        """Cập nhật ngưỡng xác suất cho mặt đường"""
        self.post_process_params['road_threshold'] = threshold
        logger.info(f"Road threshold set to {threshold}")

    def visualize(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Hiển thị kết quả phân đoạn lên frame gốc
        """
        # Tạo overlay màu xanh cho mặt đường
        overlay = frame.copy()
        overlay[mask > 0] = [0, 255, 0]  # Màu xanh lá

        # Blend với frame gốc
        result = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        # Vẽ contour mặt đường
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

        return result


# Example usage
if __name__ == "__main__":
    # Khởi tạo module
    road_seg = RoadSegmentationDeepLab(device='cpu', use_ensemble=False)  # Tắt ensemble để chạy nhanh hơn

    # Đọc frame từ camera
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Phân đoạn mặt đường
        road_mask, roi_coords = road_seg.segment_road(frame)

        # Hiển thị kết quả
        result = road_seg.visualize(frame, road_mask)

        # Hiển thị ROI
        y1, y2, x1, x2 = roi_coords
        cv2.rectangle(result, (x1, y1), (x2, y2), (255, 0, 0), 2)

        cv2.imshow('Road Segmentation', result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()