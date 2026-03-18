"""
Các hàm tiện ích mở rộng
"""

import cv2
import os
from datetime import datetime

def draw_info_panel(frame, stats, fps):
    """Vẽ bảng thông tin mở rộng"""
    h, w = frame.shape[:2]

    # Background
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (350, 200), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Vẽ thông tin
    y = 35
    cv2.putText(frame, f"FPS: {fps}", (20, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    y += 25
    cv2.putText(frame, f"Xe: {stats['total_vehicles']}", (20, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y += 20
    cv2.putText(frame, f"May: {stats['motorcycles']}", (20, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    y += 20
    cv2.putText(frame, f"Oto: {stats['cars']}", (20, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    y += 20
    cv2.putText(frame, f"VP: {stats.get('violations', 0)}", (20, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return frame

def draw_timestamp(frame):
    """Vẽ thời gian"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, timestamp, (frame.shape[1]-250, frame.shape[0]-20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return frame

def save_frame(frame, prefix='capture'):
    """Lưu frame"""
    if not os.path.exists('captures'):
        os.makedirs('captures')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"captures/{prefix}_{timestamp}.jpg"
    cv2.imwrite(filename, frame)
    return filename