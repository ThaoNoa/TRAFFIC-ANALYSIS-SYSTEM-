"""
Module theo dõi đối tượng đơn giản
"""

import cv2
import random

class SimpleTracker:
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.next_id = 0
        self.frame_count = 0

    def update(self, detections):
        """
        Cập nhật tracker với các detection mới.

        Args:
            detections: List các bounding box dạng [x1, y1, x2, y2]

        Returns:
            tracks: List các track
        """
        self.frame_count += 1

        # Dự đoán (tăng age)
        for track in self.tracks:
            track['age'] += 1
            track['time_since_update'] += 1

        # Match
        matches, unmatched_dets, unmatched_tracks = self.associate(detections)

        # Cập nhật matched tracks
        for track_idx, det_idx in matches:
            self.tracks[track_idx]['bbox'] = detections[det_idx]
            self.tracks[track_idx]['time_since_update'] = 0
            self.tracks[track_idx]['hits'] += 1
            self.tracks[track_idx]['age'] = 0

        # Tạo track mới
        for det_idx in unmatched_dets:
            track = {
                'id': self.next_id,
                'bbox': detections[det_idx],
                'hits': 1,
                'age': 0,
                'time_since_update': 0,
                'color': self._generate_color(self.next_id)
            }
            self.tracks.append(track)
            self.next_id += 1

        # Xóa track cũ
        self.tracks = [t for t in self.tracks if t['time_since_update'] <= self.max_age]

        # Lọc track
        result = []
        for track in self.tracks:
            if track['hits'] >= self.min_hits or self.frame_count <= self.min_hits:
                result.append({
                    'track_id': track['id'],
                    'bbox': track['bbox'],
                    'age': track['age'],
                    'hits': track['hits'],
                    'color': track['color']
                })

        return result

    def associate(self, detections):
        """Associate detections with existing tracks using IOU"""
        if len(self.tracks) == 0 or len(detections) == 0:
            return [], list(range(len(detections))), list(range(len(self.tracks)))

        # Tính IOU
        iou_matrix = []
        for i, det in enumerate(detections):
            row = []
            for j, track in enumerate(self.tracks):
                iou = self.calculate_iou(det, track['bbox'])
                row.append(iou)
            iou_matrix.append(row)

        matches = []
        unmatched_dets = list(range(len(detections)))
        unmatched_tracks = list(range(len(self.tracks)))

        for _ in range(min(len(detections), len(self.tracks))):
            if not unmatched_dets or not unmatched_tracks:
                break

            max_iou = 0
            max_i, max_j = -1, -1
            for i in unmatched_dets:
                for j in unmatched_tracks:
                    if iou_matrix[i][j] > max_iou:
                        max_iou = iou_matrix[i][j]
                        max_i, max_j = i, j

            if max_iou > self.iou_threshold:
                matches.append((max_j, max_i))
                unmatched_dets.remove(max_i)
                unmatched_tracks.remove(max_j)
            else:
                break

        return matches, unmatched_dets, unmatched_tracks

    @staticmethod
    def calculate_iou(box1, box2):
        """Calculate Intersection over Union between two boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0

    def _generate_color(self, track_id):
        """Generate consistent color for track ID"""
        random.seed(track_id)
        return (random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255))

    def draw_tracks(self, frame, tracks):
        """Vẽ các track lên frame"""
        for track in tracks:
            x1, y1, x2, y2 = track['bbox']
            color = track['color']
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID:{track['track_id']}", (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame