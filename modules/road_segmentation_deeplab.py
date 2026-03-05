# modules/road_segmentation_deeplab.py
"""
Module phân đoạn mặt đường sử dụng DeepLabv3+
"""

import torch
import torchvision.transforms as transforms
from torchvision import models
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

class RoadSegmentationDeepLab:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = torch.device(device)
        # Tải model DeepLabv3+ pre-trained trên COCO
        # Model này có 21 classes (background + 20 object classes). Mặt đường không phải là class riêng.
        # Để đúng với đề tài, BẠN CẦN PHẢI FINE-TUNE model này trên dataset phân đoạn mặt đường.
        # Dưới đây chỉ là cách tải và sử dụng model pre-trained để lấy ý tưởng.
        self.model = models.segmentation.deeplabv3_resnet101(pretrained=True)
        self.model = self.model.to(self.device)
        self.model.eval()
        logger.info(f"DeepLabv3+ initialized on {device}")

        # Transform cho ảnh đầu vào
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def segment_road(self, frame):
        """
        Phân đoạn mặt đường từ frame.
        Args:
            frame: Ảnh BGR đầu vào
        Returns:
            road_mask: Mask nhị phân của mặt đường (vùng 1 là đường, 0 là không)
            roi_coords: Tọa độ bounding box lớn nhất của mặt đường [y1, y2, x1, x2]
        """
        # Resize ảnh về kích thước model yêu cầu (ví dụ 512x512)
        input_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        original_h, original_w = input_image.shape[:2]
        input_tensor = self.transform(input_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)['out'][0]  # Shape: (21, H, W)
        output_predictions = output.argmax(0).byte().cpu().numpy()

        # Class 'road' không có sẵn trong COCO. Bạn cần MAP các class COCO về mặt đường.
        # Ví dụ: 'pavement'? Không có. Cách duy nhất là fine-tune.
        # Dưới đây là giả lập: coi một vài class là mặt đường (hoàn toàn không chính xác)
        road_mask = np.zeros((original_h, original_w), dtype=np.uint8)
        # Giả sử class 0 là background, 1-20 là objects. Road không nằm trong số đó.
        # -> Road mask sẽ luôn là 0. Điều này cho thấy lý do tại sao cần fine-tune.

        # --- TÌM HIỂU VỀ FINE-TUNING ---
        # Để có model phân đoạn mặt đường, bạn cần:
        # 1. Thu thập dataset ảnh giao thông có mặt đường.
        # 2. Label mặt đường (vẽ mask) bằng các công cụ như LabelMe, CVAT.
        # 3. Fine-tune model DeepLabv3+ trên dataset đó.
        # Đây là một dự án lớn, nhưng rất xứng đáng cho đề tài NCKH.

        # Tạm thời, chúng ta sẽ resize output_predictions về kích thước gốc và sử dụng nó như một mask.
        # Việc này KHÔNG ĐÚNG với đề tài nhưng để code chạy được.
        road_mask_resized = cv2.resize(output_predictions.astype(np.uint8), (original_w, original_h), interpolation=cv2.INTER_NEAREST)

        # Giả sử bất kỳ pixel nào không phải background (class 0) cũng là đường (sai)
        # road_mask = (road_mask_resized > 0).astype(np.uint8)

        # Thay vào đó, tìm contours và ưu tiên vùng trung tâm phía dưới (giả lập)
        road_mask = np.zeros((original_h, original_w), dtype=np.uint8)
        y_start = original_h * 2 // 3
        road_mask[y_start:, :] = 255  # Giả lập: phần dưới cùng là đường

        # Tìm contour lớn nhất để lấy ROI
        contours, _ = cv2.findContours(road_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            main_contour = max(contours, key=cv2.contourArea)
            x, y, w_road, h_road = cv2.boundingRect(main_contour)
            roi_coords = [y, y+h_road, x, x+w_road]
        else:
            roi_coords = [y_start, original_h, 0, original_w]

        return road_mask, roi_coords