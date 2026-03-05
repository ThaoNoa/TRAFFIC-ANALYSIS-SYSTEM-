"""
Module phân tích tư thế người điều khiển phương tiện
Sử dụng MediaPipe Pose - Phiên bản cho MediaPipe 0.10.32
"""

import cv2
import mediapipe as mp
import numpy as np
import logging

logger = logging.getLogger(__name__)

class PoseAnalyzer:
    def __init__(self):
        # MediaPipe 0.10.32 dùng cú pháp khác
        # Khởi tạo pose detection
        self.pose = mp.tasks.vision.PoseLandmarker
        self.pose_options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path='pose_landmarker.task'),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentations=False
        )

        # Tạo pose detector
        self.detector = self.pose.create_from_options(self.pose_options)

        # Drawing utils
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        logger.info("PoseAnalyzer initialized with MediaPipe Tasks API")

    def analyze(self, frame, detections):
        """
        Phân tích tư thế của người trong các bounding box.
        """
        # Chuyển ảnh sang RGB và convert to MediaPipe Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        abnormal_poses = []
        annotated_frame = frame.copy()

        try:
            # Detect pose
            detection_result = self.detector.detect_for_video(mp_image, int(cv2.getTickCount()))

            if detection_result.pose_landmarks:
                for pose_landmarks in detection_result.pose_landmarks:
                    # Vẽ khung xương
                    self._draw_landmarks(annotated_frame, pose_landmarks)

                    # Phân tích tư thế
                    result = self._analyze_pose(pose_landmarks, frame.shape)
                    if result:
                        abnormal_poses.append(result)

        except Exception as e:
            logger.error(f"Error in pose detection: {e}")

        return abnormal_poses, annotated_frame

    def _draw_landmarks(self, frame, pose_landmarks):
        """Vẽ các landmark lên frame"""
        h, w, _ = frame.shape

        # Vẽ các điểm
        for landmark in pose_landmarks:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

    def _analyze_pose(self, pose_landmarks, frame_shape):
        """Phân tích tư thế từ landmarks"""
        h, w, _ = frame_shape

        # Lấy các điểm quan trọng
        landmarks = {}
        for idx, landmark in enumerate(pose_landmarks):
            landmarks[idx] = {
                'x': landmark.x * w,
                'y': landmark.y * h,
                'visibility': landmark.visibility
            }

        # Kiểm tra nếu có đủ landmarks
        if len(landmarks) < 25:
            return None

        # Tính góc nghiêng của thân người
        try:
            left_shoulder = landmarks.get(11, {})  # LEFT_SHOULDER
            right_shoulder = landmarks.get(12, {})  # RIGHT_SHOULDER
            left_hip = landmarks.get(23, {})  # LEFT_HIP
            right_hip = landmarks.get(24, {})  # RIGHT_HIP

            if (left_shoulder and right_shoulder and left_hip and right_hip and
                left_shoulder.get('visibility', 0) > 0.5 and
                right_shoulder.get('visibility', 0) > 0.5 and
                left_hip.get('visibility', 0) > 0.5 and
                right_hip.get('visibility', 0) > 0.5):

                shoulder_center = np.array([(left_shoulder['x'] + right_shoulder['x'])/2,
                                           (left_shoulder['y'] + right_shoulder['y'])/2])
                hip_center = np.array([(left_hip['x'] + right_hip['x'])/2,
                                      (left_hip['y'] + right_hip['y'])/2])

                # Vector từ hông đến vai
                body_vector = shoulder_center - hip_center
                # Góc so với phương thẳng đứng
                angle = np.arctan2(abs(body_vector[1]), abs(body_vector[0])) * 180 / np.pi

                # Nếu góc nghiêng quá lớn
                if angle > 45:
                    return {
                        'type': 'fall_detected',
                        'angle': float(angle),
                        'confidence': float(left_shoulder.get('visibility', 0))
                    }
        except Exception as e:
            logger.error(f"Error analyzing pose: {e}")

        return None

    def __del__(self):
        """Dọn dẹp khi hủy object"""
        if hasattr(self, 'detector'):
            self.detector.close()