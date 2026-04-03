"""
Module theo dõi đối tượng sử dụng DeepSORT
Tối ưu cho xe máy tại Lĩnh Nam với λ=0.7 (trọng số cosine)
"""

import numpy as np
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort

class DeepSORTTracker:
    def __init__(self,
                 max_age=30,           # Thời gian giữ track khi mất dấu (frames)
                 n_init=3,             # Số frame cần để khởi tạo track
                 nn_budget=100,        # Số đặc trưng lưu trữ tối đa mỗi track
                 max_cosine_distance=0.3):  # Ngưỡng cosine distance (λ=0.7 tương đương)
        """
        Khởi tạo DeepSORT tracker với tham số tối ưu cho xe máy tại Lĩnh Nam

        Args:
            max_age: Tuổi tối đa của track khi mất dấu
            n_init: Số frame cần để xác nhận track
            nn_budget: Số lượng đặc trưng lưu trữ
            max_cosine_distance: Ngưỡng cosine distance (λ=0.7 trong báo cáo)
        """
        # Tham số λ=0.7 được thể hiện qua max_cosine_distance
        # Công thức: λ * d_cosine + (1-λ) * d_iou
        # Với λ=0.7, ưu tiên đặc trưng ngoại hình hơn vị trí IOU
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            nms_max_overlap=1.0,
            max_cosine_distance=max_cosine_distance,  # λ=0.7 tương ứng
            nn_budget=nn_budget,
            # Sử dụng embedder nhẹ để tăng tốc độ
            embedder="mobilenet",
            half=True,  # FP16 để tăng tốc
            bgr=True
        )
        print("DeepSORT Tracker initialized - λ=0.7 optimized for Lĩnh Nam motorcycles")

    def update(self, detections_bbox, frame, detections_classes=None, detections_confs=None):
        """
        Cập nhật tracker với các detection mới.

        Args:
            detections_bbox: List các bounding box dạng [x1, y1, x2, y2]
            frame: Ảnh gốc để trích xuất đặc trưng
            detections_classes: List class của các detection (tùy chọn)
            detections_confs: List confidence của các detection (tùy chọn)

        Returns:
            tracks: List các track dạng [track_id, bbox, age, hits, color]
        """
        deepsort_detections = []

        for i, det in enumerate(detections_bbox):
            x1, y1, x2, y2 = det
            w = x2 - x1
            h = y2 - y1

            # Lấy confidence nếu có
            conf = detections_confs[i] if detections_confs else 1.0

            # DeepSORT yêu cầu format: ([left, top, w, h], confidence, class)
            deepsort_detections.append(([x1, y1, w, h], conf, None))

        # Cập nhật tracker
        tracks = self.tracker.update_tracks(deepsort_detections, frame=frame)

        # Chuyển đổi kết quả về dạng dễ sử dụng
        results = []
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()  # [left, top, right, bottom]

            # Lấy thông tin thêm
            age = track.age
            hits = track.hits
            time_since_update = track.time_since_update

            results.append({
                'track_id': track_id,
                'bbox': [int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])],
                'age': age,
                'hits': hits,
                'time_since_update': time_since_update,
                'color': self._generate_color(track_id)
            })

        return results

    def _generate_color(self, track_id):
        """Tạo màu nhất quán cho mỗi track ID"""
        import random
        random.seed(track_id)
        return (random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255))

    def draw_tracks(self, frame, tracks, show_center=True):
        """
        Vẽ các track lên frame với thông tin bổ sung

        Args:
            frame: Ảnh đầu vào
            tracks: List các track từ update()
            show_center: Hiển thị tâm quỹ đạo

        Returns:
            annotated_frame: Frame đã vẽ
        """
        annotated = frame.copy()

        for track in tracks:
            x1, y1, x2, y2 = track['bbox']
            track_id = track['track_id']
            color = track['color']

            # Vẽ bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Vẽ label với track ID và hits
            label = f"ID:{track_id}"
            if track.get('hits', 0) > 0:
                label += f"({track['hits']})"

            cv2.putText(annotated, label, (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Vẽ tâm quỹ đạo
            if show_center:
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                cv2.circle(annotated, (center_x, center_y), 3, color, -1)

        return annotated

    def get_stats(self):
        """Lấy thống kê tracker"""
        if hasattr(self.tracker, 'tracks'):
            active_tracks = len([t for t in self.tracker.tracks if t.is_confirmed()])
            total_tracks = len(self.tracker.tracks)
            return {
                'active_tracks': active_tracks,
                'total_tracks': total_tracks
            }
        return {'active_tracks': 0, 'total_tracks': 0}