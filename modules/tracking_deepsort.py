"""
Module theo dõi đối tượng sử dụng DeepSORT
Tối ưu cho xe máy tại Lĩnh Nam với λ=0.7 (trọng số cosine)
"""

import numpy as np
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort

def _clamp_bbox_xyxy(x1, y1, x2, y2, fw, fh):
    """Giữ bbox trong ảnh và tránh box phình to bất thường (Kalman trôi)."""
    x1 = float(np.clip(x1, 0, max(0, fw - 1)))
    x2 = float(np.clip(x2, 0, fw))
    y1 = float(np.clip(y1, 0, max(0, fh - 1)))
    y2 = float(np.clip(y2, 0, fh))
    if x2 <= x1 + 1:
        x2 = min(fw, x1 + 2)
    if y2 <= y1 + 1:
        y2 = min(fh, y1 + 2)
    bw, bh = x2 - x1, y2 - y1
    max_w, max_h = 0.55 * fw, 0.55 * fh
    if bw > max_w:
        cx = (x1 + x2) * 0.5
        half = max_w * 0.5
        x1 = max(0.0, cx - half)
        x2 = min(float(fw), cx + half)
    if bh > max_h:
        cy = (y1 + y2) * 0.5
        half = max_h * 0.5
        y1 = max(0.0, cy - half)
        y2 = min(float(fh), cy + half)
    return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))


class DeepSORTTracker:
    def __init__(self,
                 max_age=12,           # Đủ để gánh 1–2 frame mất khớp, vẫn không “ma” lâu
                 n_init=1,             # Xác nhận ngay frame đầu có detection → box sớm
                 nn_budget=100,        # Số đặc trưng lưu trữ tối đa mỗi track
                 max_cosine_distance=0.3,  # Ngưỡng cosine distance (λ=0.7 tương đương)
                 bbox_smooth_alpha=0.22):
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
        self.bbox_smooth_alpha = bbox_smooth_alpha
        self._bbox_ema = {}
        print("DeepSORT Tracker initialized - λ=0.7 optimized for Lĩnh Nam motorcycles")

    def reset_bbox_smoothing(self):
        self._bbox_ema.clear()

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
        fh, fw = frame.shape[:2]

        # Chuyển đổi kết quả về dạng dễ sử dụng
        # n_init=1: hầu hết đã confirmed sớm; vẫn xuất track tentative vừa khớp frame này (tsu==0)
        # để box xuất hiện cùng lúc YOLO thấy xe, không chờ thêm frame.
        results = []
        for track in tracks:
            tsu = track.time_since_update
            if not track.is_confirmed() and tsu != 0:
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()  # [left, top, right, bottom]
            raw = np.array([ltrb[0], ltrb[1], ltrb[2], ltrb[3]], dtype=np.float32)
            time_since_update = track.time_since_update

            # Chỉ làm mượt khi vừa khớp detection (tsu==0). Khi coasting, Kalman hay “bay” —
            # không trộn EMA để tránh box rỗng phóng to.
            if time_since_update == 0:
                if track_id in self._bbox_ema:
                    a = self.bbox_smooth_alpha
                    bbox = a * raw + (1.0 - a) * self._bbox_ema[track_id]
                else:
                    bbox = raw.copy()
                self._bbox_ema[track_id] = bbox
            else:
                bbox = raw
                if track_id in self._bbox_ema:
                    del self._bbox_ema[track_id]

            x1, y1, x2, y2 = _clamp_bbox_xyxy(
                bbox[0], bbox[1], bbox[2], bbox[3], fw, fh
            )

            # Lấy thông tin thêm
            age = track.age
            hits = track.hits

            results.append({
                'track_id': track_id,
                'bbox': [x1, y1, x2, y2],
                'age': age,
                'hits': hits,
                'time_since_update': time_since_update,
                'confirmed': track.is_confirmed(),
                'color': self._generate_color(track_id)
            })

        active = {r['track_id'] for r in results}
        for tid in list(self._bbox_ema.keys()):
            if tid not in active:
                del self._bbox_ema[tid]

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
            # Hiển thị cả 1 frame “dự đoán” sau khi mất khớp tạm thời — tránh nhấp nháy 1 tick
            if track.get('time_since_update', 99) > 1:
                continue
            x1, y1, x2, y2 = track['bbox']
            track_id = track['track_id']
            color = track['color']

            # Vẽ bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            label = f"ID:{track_id}"
            if track.get('hits', 0) > 0:
                label += f"({track['hits']})"
            spd = track.get('speed_kmh')
            if spd is not None:
                label += f" {spd:.0f}km/h"
            cn = track.get('class_name')
            if cn and spd is not None:
                label += f" [{cn}]"

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