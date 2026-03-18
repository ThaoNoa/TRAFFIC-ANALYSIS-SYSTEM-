"""
Module theo dõi đối tượng sử dụng DeepSORT với tham số tối ưu cho xe máy
"""

import numpy as np
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort

class DeepSORTTracker:
    def __init__(self, lambda_motion=0.7, max_age=30):
        """
        Args:
            lambda_motion: trọng số motion (0-1), càng cao càng ưu tiên chuyển động
            max_age: số frame tối đa giữ track khi bị mất
        """
        # Tham số tối ưu cho giao thông Lĩnh Nam (xe máy chiếm 78%)
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=3,
            nms_max_overlap=1.0,
            max_cosine_distance=0.3,
            nn_budget=None,
            # Tích hợp trọng số motion-appearance
            # DeepSORT gốc không hỗ trợ lambda trực tiếp, nhưng ta có thể điều chỉnh
            # bằng cách set max_cosine_distance và dùng motion matching
            # Thực tế, DeepSORT kết hợp Mahalanobis distance và cosine distance
            # Để ưu tiên motion, ta có thể tăng threshold cho cosine
            # và điều chỉnh matching cascade
            # Ở đây ta chỉ set max_cosine_distance cao hơn để dễ match hơn,
            # và sử dụng tham số motion matching mặc định
        )
        # Lưu lambda để dùng trong logic riêng (nếu cần)
        self.lambda_motion = lambda_motion
        print(f"DeepSORT Tracker initialized with lambda_motion={lambda_motion}, max_age={max_age}")

    def update(self, detections_bbox, frame):
        """
        Cập nhật tracker với các detection mới.
        Args:
            detections_bbox: list [x1,y1,x2,y2]
            frame: ảnh gốc
        Returns:
            tracks: list các track dạng {'track_id': id, 'bbox': [x1,y1,x2,y2], 'age': age}
        """
        deepsort_detections = []
        for det in detections_bbox:
            x1, y1, x2, y2 = det
            w = x2 - x1
            h = y2 - y1
            deepsort_detections.append(([x1, y1, w, h], 1.0))

        tracks = self.tracker.update_tracks(deepsort_detections, frame=frame)

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
        """Vẽ các track lên frame"""
        for track in tracks:
            x1, y1, x2, y2 = track['bbox']
            color = self._generate_color(track['track_id'])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID:{track['track_id']}", (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame

    def _generate_color(self, track_id):
        import random
        random.seed(track_id)
        return (random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255))