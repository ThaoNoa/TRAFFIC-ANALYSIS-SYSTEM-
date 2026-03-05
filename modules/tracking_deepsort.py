# modules/tracking_deepsort.py
"""
Module theo dõi đối tượng sử dụng DeepSORT
"""

import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

class DeepSORTTracker:
    def __init__(self):
        # Khởi tạo DeepSORT với các tham số phù hợp
        # Bạn có thể cần điều chỉnh max_age, n_init, max_cosine_distance...
        self.tracker = DeepSort(
            max_age=30,
            n_init=3,
            nms_max_overlap=1.0,
            max_cosine_distance=0.3,
            nn_budget=None,
            # Cần chỉ định đường dẫn đến model feature extractor
            # model_filename='path/to/your/feature_extractor.pb',
            # Nếu không có file model, DeepSORT sẽ chạy ở chế độ SORT thuần túy.
            # Để đúng với đề tài, bạn cần tải model feature extractor (thường là cosine metric learning model)
        )
        print("DeepSORT Tracker initialized")

    def update(self, detections_bbox, frame):
        """
        Cập nhật tracker với các detection mới.
        Args:
            detections_bbox: List các bounding box dạng [x1, y1, x2, y2]
            frame: Ảnh gốc để trích xuất đặc trưng (cần cho DeepSORT)
        Returns:
            tracks: List các track dạng [track_id, bbox, confidence, ...]
        """
        # DeepSORT yêu cầu input là list các detection dạng
        # ([left, top, w, h], confidence, feature)
        # Hoặc ([left, top, w, h], confidence)
        deepsort_detections = []
        for det in detections_bbox:
            x1, y1, x2, y2 = det
            w = x2 - x1
            h = y2 - y1
            # Giả sử confidence = 1.0 nếu không có. Bạn có thể lấy từ detector.
            deepsort_detections.append(([x1, y1, w, h], 1.0))

        tracks = self.tracker.update_tracks(deepsort_detections, frame=frame)

        # Chuyển đổi kết quả về dạng dễ sử dụng
        results = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()  # [left, top, right, bottom]
            results.append({
                'track_id': track_id,
                'bbox': [int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])],
                'age': track.age
            })
        return results

    def draw_tracks(self, frame, tracks):
        """Vẽ các track lên frame (giống hàm cũ)"""
        # Giữ nguyên hàm vẽ từ SimpleTracker của bạn, nó rất tốt
        for track in tracks:
            x1, y1, x2, y2 = track['bbox']
            # Màu sắc có thể dựa trên track_id để nhất quán
            color = self._generate_color(track['track_id'])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID:{track['track_id']}", (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame

    def _generate_color(self, track_id):
        # Giữ nguyên hàm generate_color từ SimpleTracker
        import random
        random.seed(track_id)
        return (random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255))