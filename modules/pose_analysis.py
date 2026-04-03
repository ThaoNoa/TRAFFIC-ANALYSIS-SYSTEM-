"""
Module phân tích tư thế người điều khiển phương tiện
Sử dụng MediaPipe Pose - Trích xuất 18 điểm khớp xương theo chuẩn COCO
Tối ưu cho CPU/GPU với FP16
"""

import cv2
import numpy as np
import mediapipe as mp
import logging

logger = logging.getLogger(__name__)

class PoseAnalyzer:
    """
    Phân tích tư thế người lái xe
    Trích xuất 18 điểm khớp xương (COCO format)
    """

    # Mapping từ MediaPipe 33 landmarks sang 18 keypoints COCO
    # COCO keypoint indices:
    # 0: nose, 1: L eye, 2: R eye, 3: L ear, 4: R ear,
    # 5: L shoulder, 6: R shoulder, 7: L elbow, 8: R elbow,
    # 9: L wrist, 10: R wrist, 11: L hip, 12: R hip,
    # 13: L knee, 14: R knee, 15: L ankle, 16: R ankle, 17: neck

    MEDIAPIPE_TO_COCO = {
        # Mặt
        0: 0,   # nose -> nose
        2: 1,   # left_eye_outer -> L eye (gần đúng)
        5: 2,   # right_eye_outer -> R eye (gần đúng)
        7: 3,   # left_ear -> L ear
        8: 4,   # right_ear -> R ear
        # Vai
        11: 5,  # left_shoulder -> L shoulder
        12: 6,  # right_shoulder -> R shoulder
        # Khuỷu tay
        13: 7,  # left_elbow -> L elbow
        14: 8,  # right_elbow -> R elbow
        # Cổ tay
        15: 9,  # left_wrist -> L wrist
        16: 10, # right_wrist -> R wrist
        # Hông
        23: 11, # left_hip -> L hip
        24: 12, # right_hip -> R hip
        # Đầu gối
        25: 13, # left_knee -> L knee
        26: 14, # right_knee -> R knee
        # Mắt cá
        27: 15, # left_ankle -> L ankle
        28: 16, # right_ankle -> R ankle
        # Cổ (ước lượng từ vai)
        'neck': 17  # neck - tính toán từ vai
    }

    # Các cặp keypoints để vẽ khung xương (skeleton)
    SKELETON = [
        (0, 1), (0, 2),           # Mũi - mắt
        (1, 3), (2, 4),           # Mắt - tai
        (5, 6),                   # Vai - vai (ngang)
        (5, 7), (6, 8),           # Vai - khuỷu tay
        (7, 9), (8, 10),          # Khuỷu tay - cổ tay
        (5, 11), (6, 12),         # Vai - hông
        (11, 12),                 # Hông - hông (ngang)
        (11, 13), (12, 14),       # Hông - đầu gối
        (13, 15), (14, 16),       # Đầu gối - mắt cá
        (5, 17), (6, 17)          # Vai - cổ
    ]

    def __init__(self,
                 model_complexity=1,      # 0,1,2 (2 là chính xác nhất nhưng chậm)
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5,
                 use_fp16=True):
        """
        Khởi tạo PoseAnalyzer với MediaPipe

        Args:
            model_complexity: Độ phức tạp model (0=light, 1=medium, 2=full)
            min_detection_confidence: Ngưỡng confidence phát hiện
            min_tracking_confidence: Ngưỡng confidence tracking
            use_fp16: Sử dụng FP16 để tăng tốc
        """
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        # Khởi tạo MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        # Drawing utils
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Màu sắc cho skeleton
        self.skeleton_colors = {
            'face': (255, 255, 0),      # Vàng
            'arm': (0, 255, 0),         # Xanh lá
            'leg': (0, 255, 255),       # Cyan
            'body': (0, 165, 255),      # Cam
            'joint': (0, 0, 255)        # Đỏ
        }

        logger.info(f"PoseAnalyzer initialized - model_complexity={model_complexity}")

    def extract_18_keypoints(self, pose_landmarks, h, w):
        """
        Trích xuất 18 keypoints từ MediaPipe landmarks

        Args:
            pose_landmarks: MediaPipe pose landmarks
            h, w: Kích thước frame

        Returns:
            keypoints: Mảng 18x3 (x, y, confidence)
        """
        # Khởi tạo mảng keypoints (x, y, visibility)
        keypoints = np.zeros((18, 3), dtype=np.float32)

        if not pose_landmarks:
            return keypoints

        landmarks_dict = {}
        for idx, landmark in enumerate(pose_landmarks.landmark):
            landmarks_dict[idx] = {
                'x': landmark.x * w,
                'y': landmark.y * h,
                'visibility': landmark.visibility,
                'presence': landmark.presence
            }

        # Map MediaPipe landmarks sang COCO keypoints
        for mp_idx, coco_idx in self.MEDIAPIPE_TO_COCO.items():
            if mp_idx == 'neck':
                # Neck = trung điểm giữa 2 vai
                if 11 in landmarks_dict and 12 in landmarks_dict:
                    left_shoulder = landmarks_dict[11]
                    right_shoulder = landmarks_dict[12]
                    keypoints[17, 0] = (left_shoulder['x'] + right_shoulder['x']) / 2
                    keypoints[17, 1] = (left_shoulder['y'] + right_shoulder['y']) / 2
                    keypoints[17, 2] = min(left_shoulder['visibility'], right_shoulder['visibility'])
            elif mp_idx in landmarks_dict:
                keypoints[coco_idx, 0] = landmarks_dict[mp_idx]['x']
                keypoints[coco_idx, 1] = landmarks_dict[mp_idx]['y']
                keypoints[coco_idx, 2] = landmarks_dict[mp_idx]['visibility']

        return keypoints

    def analyze(self, frame, detections):
        """
        Phân tích tư thế của người trong frame

        Args:
            frame: Ảnh đầu vào (BGR)
            detections: List các detection từ YOLO

        Returns:
            abnormal_poses: List các tư thế bất thường phát hiện được
            annotated_frame: Frame đã vẽ kết quả
        """
        # Chuyển sang RGB cho MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        annotated_frame = frame.copy()

        # Process pose
        results = self.pose.process(frame_rgb)

        abnormal_poses = []

        if results.pose_landmarks:
            h, w = frame.shape[:2]

            # Trích xuất 18 keypoints
            keypoints = self.extract_18_keypoints(results.pose_landmarks, h, w)

            # Vẽ khung xương
            annotated_frame = self.draw_skeleton(annotated_frame, keypoints)

            # Phân tích tư thế
            pose_analysis = self._analyze_pose_angles(keypoints)

            if pose_analysis['is_abnormal']:
                abnormal_poses.append({
                    'type': pose_analysis['type'],
                    'angle': pose_analysis['angle'],
                    'confidence': pose_analysis['confidence'],
                    'keypoints': keypoints.tolist()
                })

                # Vẽ cảnh báo
                cv2.putText(annotated_frame,
                           f"⚠️ {pose_analysis['message']}",
                           (50, 50),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.7, (0, 0, 255), 2)

            # Vẽ các keypoints
            annotated_frame = self.draw_keypoints(annotated_frame, keypoints)

        return abnormal_poses, annotated_frame

    def draw_keypoints(self, frame, keypoints, radius=4):
        """
        Vẽ các keypoints lên frame

        Args:
            frame: Ảnh đầu vào
            keypoints: Mảng 18x3 (x, y, confidence)
            radius: Bán kính điểm
        """
        for i, kp in enumerate(keypoints):
            x, y, conf = kp
            if conf > 0.5 and x > 0 and y > 0:
                cv2.circle(frame, (int(x), int(y)), radius,
                          self.skeleton_colors['joint'], -1)
                # Vẽ số thứ tự keypoint (debug)
                # cv2.putText(frame, str(i), (int(x)-5, int(y)-5),
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)
        return frame

    def draw_skeleton(self, frame, keypoints):
        """
        Vẽ khung xương từ các keypoints

        Args:
            frame: Ảnh đầu vào
            keypoints: Mảng 18x3 (x, y, confidence)
        """
        for (start, end) in self.SKELETON:
            if start < len(keypoints) and end < len(keypoints):
                x1, y1, conf1 = keypoints[start]
                x2, y2, conf2 = keypoints[end]

                if conf1 > 0.5 and conf2 > 0.5:
                    # Chọn màu dựa trên vị trí
                    if start <= 4:
                        color = self.skeleton_colors['face']
                    elif start <= 10:
                        color = self.skeleton_colors['arm']
                    elif start <= 16:
                        color = self.skeleton_colors['leg']
                    else:
                        color = self.skeleton_colors['body']

                    cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        return frame

    def _analyze_pose_angles(self, keypoints):
        """
        Phân tích góc giữa các khớp để phát hiện tư thế bất thường

        Returns:
            dict: Kết quả phân tích
        """
        result = {
            'is_abnormal': False,
            'type': 'normal',
            'angle': 0,
            'confidence': 0,
            'message': ''
        }

        # Lấy các keypoints cần thiết
        left_shoulder = keypoints[5]  # L shoulder
        right_shoulder = keypoints[6]  # R shoulder
        left_hip = keypoints[11]        # L hip
        right_hip = keypoints[12]       # R hip
        left_knee = keypoints[13]       # L knee
        right_knee = keypoints[14]      # R knee
        left_ankle = keypoints[15]      # L ankle
        right_ankle = keypoints[16]     # R ankle
        neck = keypoints[17]            # Neck

        # Kiểm tra độ tin cậy
        if neck[2] < 0.3:
            return result

        # === 1. TÍNH GÓC NGHIÊNG CỦA THÂN NGƯỜI ===
        shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) / 2
        shoulder_center_y = (left_shoulder[1] + right_shoulder[1]) / 2
        hip_center_x = (left_hip[0] + right_hip[0]) / 2
        hip_center_y = (left_hip[1] + right_hip[1]) / 2

        # Vector từ hông đến vai
        body_vector = np.array([shoulder_center_x - hip_center_x,
                                shoulder_center_y - hip_center_y])

        # Vector thẳng đứng
        vertical_vector = np.array([0, 1])

        # Tính góc nghiêng (độ)
        if np.linalg.norm(body_vector) > 0:
            dot = np.dot(body_vector, vertical_vector)
            norm = np.linalg.norm(body_vector)
            angle = np.arccos(np.clip(dot / norm, -1, 1)) * 180 / np.pi
        else:
            angle = 0

        # === 2. PHÁT HIỆN TƯ THẾ BẤT THƯỜNG ===

        # 2.1 Người nằm/ngã (góc nghiêng > 45 độ)
        if angle > 45:
            result['is_abnormal'] = True
            result['type'] = 'fall_detected'
            result['angle'] = float(angle)
            result['confidence'] = float(neck[2])
            result['message'] = f'PHAT HIEN NGA XE! Goc nghieng: {angle:.1f}°'
            return result

        # 2.2 Người cúi gập (góc giữa đùi và thân)
        if left_hip[2] > 0.5 and left_knee[2] > 0.5:
            thigh_vector = np.array([left_knee[0] - left_hip[0],
                                     left_knee[1] - left_hip[1]])
            torso_vector = np.array([shoulder_center_x - hip_center_x,
                                     shoulder_center_y - hip_center_y])

            if np.linalg.norm(thigh_vector) > 0 and np.linalg.norm(torso_vector) > 0:
                dot = np.dot(thigh_vector, torso_vector)
                norm_product = np.linalg.norm(thigh_vector) * np.linalg.norm(torso_vector)
                hip_angle = np.arccos(np.clip(dot / norm_product, -1, 1)) * 180 / np.pi

                # Góc hông < 60 độ là cúi gập
                if hip_angle < 60:
                    result['is_abnormal'] = True
                    result['type'] = 'bending_over'
                    result['angle'] = float(hip_angle)
                    result['confidence'] = float(left_hip[2])
                    result['message'] = f'CUOI NGUOI! Goc hong: {hip_angle:.1f}°'
                    return result

        # 2.3 Giơ tay (góc khuỷu tay < 90 độ)
        left_elbow = keypoints[7]
        right_elbow = keypoints[8]
        left_wrist = keypoints[9]
        right_wrist = keypoints[10]

        for elbow, wrist, side in [(left_elbow, left_wrist, 'trai'),
                                    (right_elbow, right_wrist, 'phai')]:
            if elbow[2] > 0.5 and wrist[2] > 0.5:
                arm_vector = np.array([wrist[0] - elbow[0], wrist[1] - elbow[1]])
                if np.linalg.norm(arm_vector) > 0:
                    # Góc với phương ngang
                    arm_angle = np.arctan2(arm_vector[1], arm_vector[0]) * 180 / np.pi
                    # Nếu giơ tay lên (góc dương lớn)
                    if arm_angle > 30:
                        result['is_abnormal'] = True
                        result['type'] = 'hand_raise'
                        result['angle'] = float(arm_angle)
                        result['confidence'] = float(elbow[2])
                        result['message'] = f'GIO TAY {side.upper()}! Goc: {arm_angle:.1f}°'
                        return result

        return result

    def get_keypoints_angles(self, keypoints):
        """
        Tính toán các góc quan trọng từ keypoints

        Returns:
            dict: Các góc của khớp
        """
        angles = {}

        # Góc vai - khuỷu tay - cổ tay
        left_shoulder = keypoints[5]
        left_elbow = keypoints[7]
        left_wrist = keypoints[9]

        if all(kp[2] > 0.5 for kp in [left_shoulder, left_elbow, left_wrist]):
            angles['left_arm'] = self._calculate_angle(
                left_shoulder[:2], left_elbow[:2], left_wrist[:2]
            )

        right_shoulder = keypoints[6]
        right_elbow = keypoints[8]
        right_wrist = keypoints[10]

        if all(kp[2] > 0.5 for kp in [right_shoulder, right_elbow, right_wrist]):
            angles['right_arm'] = self._calculate_angle(
                right_shoulder[:2], right_elbow[:2], right_wrist[:2]
            )

        # Góc hông - đầu gối - mắt cá
        left_hip = keypoints[11]
        left_knee = keypoints[13]
        left_ankle = keypoints[15]

        if all(kp[2] > 0.5 for kp in [left_hip, left_knee, left_ankle]):
            angles['left_leg'] = self._calculate_angle(
                left_hip[:2], left_knee[:2], left_ankle[:2]
            )

        right_hip = keypoints[12]
        right_knee = keypoints[14]
        right_ankle = keypoints[16]

        if all(kp[2] > 0.5 for kp in [right_hip, right_knee, right_ankle]):
            angles['right_leg'] = self._calculate_angle(
                right_hip[:2], right_knee[:2], right_ankle[:2]
            )

        return angles

    def _calculate_angle(self, p1, p2, p3):
        """Tính góc giữa 3 điểm"""
        v1 = np.array(p1) - np.array(p2)
        v2 = np.array(p3) - np.array(p2)

        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
            return 0

        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi
        return float(angle)

    def __del__(self):
        """Dọn dẹp khi hủy object"""
        if hasattr(self, 'pose'):
            self.pose.close()