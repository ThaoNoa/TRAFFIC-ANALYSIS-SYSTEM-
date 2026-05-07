"""
Hệ thống phân tích giao thông Lĩnh Nam - HỖ TRỢ CAMERA TRỰC TIẾP
Tích hợp: YOLOv8 (phát hiện), DeepSORT (theo dõi),
Pose Analysis, Road Analysis
"""

import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import queue
import time
from datetime import datetime
import os
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.road_analysis import RoadAnalyzer
from modules.detection import VehicleDetector
from modules.tracking_deepsort import DeepSORTTracker
from modules.road_integrator import RoadIntegrator
from modules.pose_analysis_simple import PoseAnalyzer
from modules.violation_detector import ViolationDetector
from modules.utils import best_detection_for_track
from modules.ipm import IPMTransformer, VehicleTrackerWithIPM


class TrafficAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hệ thống phân tích giao thông Lĩnh Nam - AI Pro (Camera trực tiếp)")
        self.root.geometry("1600x900")
        self.road_analyzer = RoadAnalyzer()

        self.camera_id = 0
        self.camera_width = 640
        self.camera_height = 480
        self.camera_connected = False

        print("=" * 70)
        print("KHỞI TẠO HỆ THỐNG PHÂN TÍCH GIAO THÔNG THÔNG MINH")
        print("=" * 70)

        try:
            print("1. Khởi tạo VehicleDetector (YOLOv8)...")
            self.detector = VehicleDetector()

            print("2. Khởi tạo DeepSORTTracker...")
            self.tracker = DeepSORTTracker()

            print("3. Khởi tạo RoadIntegrator...")
            self.road_integrator = RoadIntegrator()

            print("4. Khởi tạo PoseAnalyzer (Simple)...")
            self.pose_analyzer = PoseAnalyzer()

            print("5. Khởi tạo ViolationDetector...")
            self.violation_detector = ViolationDetector()

            print("6. Khởi tạo IPM + tốc độ theo track...")
            self.ipm = IPMTransformer(frame_width=self.camera_width, frame_height=self.camera_height)
            self.speed_tracker = VehicleTrackerWithIPM(self.ipm)

            print("✅ Đã khởi tạo xong tất cả modules!")
        except Exception as e:
            print(f"❌ Lỗi khởi tạo module: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Lỗi", f"Không thể khởi tạo module: {e}")
            sys.exit(1)

        self.video_path = None
        self.cap = None
        self.is_running = False
        self.is_paused = False
        self.analysis_mode = "none"
        self.frame_queue = queue.Queue(maxsize=5)
        self.result_queue = queue.Queue(maxsize=5)

        self.total_frames = 0
        self.current_frame = 0
        self.fps_original = 30
        self.frame_duration = 1.0 / 30

        self.create_ui()

        self.fps_display = 0
        self.frame_count = 0
        self.last_time = time.time()

        self.stats = {
            'total_vehicles': 0, 'motorcycles': 0, 'cars': 0, 'trucks': 0,
            'buses': 0, 'persons': 0, 'bicycles': 0, 'violations': 0,
            'no_helmet': 0, 'speeding': 0, 'obstacles': 0
        }

        self.counted_track_ids = set()

        self._vehicle_classes_speed = frozenset({'xe_may', 'xe_oto', 'xe_bus', 'xe_tai', 'xe_dap'})
        self.violations_log = []

        print("=" * 70)
        print("🟢 HỆ THỐNG SẴN SÀNG")
        print("=" * 70)

    def _is_new_vehicle(self, center_x, center_y, current_frame):
        """
        Kiểm tra xe có phải mới (chưa đếm) không.
        Dùng lưới để gom các vị trí gần nhau.
        """
        grid_x = center_x // self.position_grid_size
        grid_y = center_y // self.position_grid_size
        key = (grid_x, grid_y)

        if key not in self.seen_vehicles:
            # Chưa từng thấy xe ở ô này
            self.seen_vehicles[key] = current_frame
            return True
        else:
            last_seen = self.seen_vehicles[key]
            # Nếu đã lâu không thấy xe ở ô này (hoặc xe đã rời đi), cho đếm lại
            if current_frame - last_seen > self.seen_frames_threshold:
                self.seen_vehicles[key] = current_frame
                return True
            return False

    def create_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=4)
        main_frame.columnconfigure(1, weight=2)
        main_frame.rowconfigure(0, weight=1)

        left_frame = ttk.Frame(main_frame)
        left_frame.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        left_frame.columnconfigure(0, weight=1)
        left_frame.rowconfigure(0, weight=1)

        self.video_label = ttk.Label(left_frame, relief=tk.SUNKEN, background='black')
        self.video_label.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))

        control_frame = ttk.Frame(left_frame)
        control_frame.grid(row=1, column=0, pady=10)

        ttk.Label(control_frame, text="Camera ID:").grid(row=0, column=0, padx=2)
        self.camera_combo = ttk.Combobox(control_frame, values=["0", "1", "2", "3"], width=5)
        self.camera_combo.grid(row=0, column=1, padx=2)
        self.camera_combo.set("0")

        ttk.Button(control_frame, text="📷 Mở Camera", command=self.open_camera).grid(row=0, column=2, padx=5)
        ttk.Button(control_frame, text="📁 Mở Video", command=self.open_video).grid(row=0, column=3, padx=5)

        # === HAI NÚT PHÂN TÍCH ===
        btn_full = ttk.Button(control_frame, text="🛣️ PHÂN TÍCH (Trừ phương tiện)",
                              command=self.start_analysis_full)
        btn_full.grid(row=1, column=0, columnspan=4, padx=5, pady=5)

        btn_vehicle = ttk.Button(control_frame, text="🚗 CHỈ PHÂN TÍCH PHƯƠNG TIỆN",
                                 command=self.start_analysis_vehicle_only)
        btn_vehicle.grid(row=1, column=4, columnspan=4, padx=5, pady=5)

        ttk.Button(control_frame, text="⏸ Tạm dừng", command=self.toggle_pause).grid(row=2, column=0, padx=5, pady=5)
        ttk.Button(control_frame, text="⏹ Dừng", command=self.stop_analysis).grid(row=2, column=1, padx=5, pady=5)
        ttk.Button(control_frame, text="📸 Chụp ảnh", command=self.capture_image).grid(row=2, column=2, padx=5, pady=5)

        self.mode_label = ttk.Label(control_frame, text="⚙️ Chế độ: Chưa chọn", font=('Arial', 10, 'bold'))
        self.mode_label.grid(row=3, column=0, columnspan=8, pady=5)

        info_frame = ttk.Frame(left_frame)
        info_frame.grid(row=2, column=0, pady=5, sticky=(tk.W, tk.E))

        self.video_info = ttk.Label(info_frame, text="Chưa kết nối camera", font=('Arial', 10))
        self.video_info.grid(row=0, column=0, sticky=tk.W)

        self.fps_label = ttk.Label(info_frame, text="FPS: 0", font=('Arial', 10))
        self.fps_label.grid(row=0, column=1, padx=20)

        self.time_label = ttk.Label(info_frame, text="Time: --:--", font=('Arial', 10))
        self.time_label.grid(row=0, column=2, padx=20)

        self.progress_bar = ttk.Progressbar(left_frame, orient=tk.HORIZONTAL, length=800, mode='determinate')
        self.progress_bar.grid(row=3, column=0, pady=5, sticky=(tk.W, tk.E))
        self.progress_bar.configure(mode='indeterminate')

        right_frame = ttk.Frame(main_frame, width=600)
        right_frame.grid(row=0, column=1, padx=5, pady=5, sticky=(tk.N, tk.S, tk.W, tk.E))
        right_frame.grid_propagate(False)
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(0, weight=1)

        notebook = ttk.Notebook(right_frame)
        notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        stats_frame = ttk.Frame(notebook)
        notebook.add(stats_frame, text="📊 Thống kê")
        stats_frame.columnconfigure(0, weight=1)
        stats_frame.rowconfigure(0, weight=1)

        self.stats_text = tk.Text(stats_frame, width=70, height=35, font=('Courier', 9))
        stats_scroll = ttk.Scrollbar(stats_frame, orient="vertical", command=self.stats_text.yview)
        self.stats_text.configure(yscrollcommand=stats_scroll.set)
        self.stats_text.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        stats_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))

        road_frame = ttk.Frame(notebook)
        notebook.add(road_frame, text="🛣️ Mặt đường")
        road_frame.columnconfigure(0, weight=1)
        road_frame.rowconfigure(0, weight=1)

        self.road_text = tk.Text(road_frame, width=70, height=35, font=('Courier', 9))
        road_scroll = ttk.Scrollbar(road_frame, orient="vertical", command=self.road_text.yview)
        self.road_text.configure(yscrollcommand=road_scroll.set)
        self.road_text.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        road_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))

        violation_frame = ttk.Frame(notebook)
        notebook.add(violation_frame, text="🚨 Vi phạm")
        violation_frame.columnconfigure(0, weight=1)
        violation_frame.rowconfigure(0, weight=1)

        self.violation_text = tk.Text(violation_frame, width=70, height=35, font=('Courier', 9))
        violation_scroll = ttk.Scrollbar(violation_frame, orient="vertical", command=self.violation_text.yview)
        self.violation_text.configure(yscrollcommand=violation_scroll.set)
        self.violation_text.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        violation_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))

        log_frame = ttk.Frame(notebook)
        notebook.add(log_frame, text="📝 Log")
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        self.log_text = tk.Text(log_frame, width=70, height=35, font=('Courier', 9))
        log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        self.log_text.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        log_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))

        self.status_bar = ttk.Label(self.root, text="Sẵn sàng - Mở camera và chọn chế độ phân tích", relief=tk.SUNKEN)
        self.status_bar.grid(row=1, column=0, sticky=(tk.W, tk.E))

        self.log("🚀 Hệ thống AI phân tích giao thông đã khởi động")
        self.log("📷 Nhấn 'Mở Camera' để kết nối, sau đó chọn chế độ phân tích")

    def start_analysis_full(self):
        """Phân tích TẤT CẢ TRỪ PHƯƠNG TIỆN (mặt đường, chướng ngại vật, vi phạm, pose)"""
        if not self.cap or not self.cap.isOpened():
            messagebox.showwarning("Cảnh báo", "Vui lòng mở camera hoặc video trước!")
            return
        self.analysis_mode = "full"
        self.mode_label.config(text="⚙️ Chế độ: PHÂN TÍCH (Trừ phương tiện)", foreground="green")
        self.log("🛣️ Bắt đầu phân tích - Bao gồm mặt đường, ổ gà, chướng ngại vật (KHÔNG phân tích phương tiện)")
        self.start_analysis()

    def start_analysis_vehicle_only(self):
        """Chỉ phân tích phương tiện và theo dõi"""
        if not self.cap or not self.cap.isOpened():
            messagebox.showwarning("Cảnh báo", "Vui lòng mở camera hoặc video trước!")
            return
        self.analysis_mode = "vehicle_only"
        self.mode_label.config(text="⚙️ Chế độ: CHỈ PHÂN TÍCH PHƯƠNG TIỆN", foreground="blue")
        self.log("🚗 Bắt đầu phân tích chế độ CHỈ PHƯƠNG TIỆN")
        self.start_analysis()

    def open_camera(self):
        if self.is_running:
            self.stop_analysis()

        if self.cap is not None:
            self.cap.release()

        selected_id = int(self.camera_combo.get())
        self.camera_connected = False

        for cam_id in [selected_id, 0, 1, 2]:
            cap = cv2.VideoCapture(cam_id)
            if cap.isOpened():
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    self.cap = cap
                    self.camera_id = cam_id
                    self.camera_combo.set(str(cam_id))
                    self.camera_connected = True
                    break
                else:
                    cap.release()

        if not self.camera_connected:
            messagebox.showerror("Lỗi", "Không thể kết nối camera!")
            self.log("❌ Không thể kết nối camera")
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)

        self.fps_original = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps_original <= 0 or self.fps_original > 60:
            self.fps_original = 30
        self.frame_duration = 1.0 / self.fps_original

        self.video_info.config(text=f"📷 CAMERA ID {self.camera_id} | ~{self.fps_original:.1f}fps")
        self.time_label.config(text="Time: Live Camera")
        self.progress_bar.configure(mode='indeterminate')
        self.status_bar.config(text=f"Đã kết nối camera ID {self.camera_id}")
        self.log(f"✅ Đã kết nối camera ID {self.camera_id}")
        self.log("👉 Hãy chọn chế độ phân tích để bắt đầu")

    def open_video(self):
        file_path = filedialog.askopenfilename(
            title="Chọn video giao thông",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )

        if file_path:
            if self.is_running:
                self.stop_analysis()

            if self.cap is not None:
                self.cap.release()

            self.video_path = file_path
            self.cap = cv2.VideoCapture(file_path)

            if self.cap.isOpened():
                self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.fps_original = self.cap.get(cv2.CAP_PROP_FPS)
                if self.fps_original <= 0:
                    self.fps_original = 30
                self.frame_duration = 1.0 / self.fps_original

                width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                duration = self.total_frames / self.fps_original
                minutes = int(duration // 60)
                seconds = int(duration % 60)

                self.video_info.config(text=f"📁 {os.path.basename(file_path)} | {width}x{height} | {minutes:02d}:{seconds:02d}")
                self.progress_bar['maximum'] = self.total_frames
                self.progress_bar.configure(mode='determinate')
                self.progress_bar['value'] = 0
                self.current_frame = 0

                self.status_bar.config(text=f"Đã chọn: {os.path.basename(file_path)}")
                self.log(f"📁 Đã chọn video: {os.path.basename(file_path)}")
                self.log("👉 Hãy chọn chế độ phân tích để bắt đầu")
            else:
                messagebox.showerror("Lỗi", "Không thể mở file video!")

    def start_analysis(self):
        if self.analysis_mode == "none":
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn chế độ phân tích trước!")
            return

        if self.cap is None or not self.cap.isOpened():
            if self.video_path:
                self.cap = cv2.VideoCapture(self.video_path)
                if not self.cap.isOpened():
                    messagebox.showerror("Lỗi", "Không thể mở video!")
                    return
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.current_frame = 0
                self.progress_bar.configure(mode='determinate')
                self.progress_bar['value'] = 0
            else:
                messagebox.showwarning("Cảnh báo", "Vui lòng mở camera hoặc chọn video trước!")
                return

        self.stats = {
            'total_vehicles': 0, 'motorcycles': 0, 'cars': 0, 'trucks': 0,
            'buses': 0, 'persons': 0, 'bicycles': 0, 'violations': 0,
            'no_helmet': 0, 'speeding': 0, 'wrong_lane': 0, 'obstacles': 0
        }

        self.counted_track_ids.clear()

        self.violations_log = []
        self.counted_track_ids.clear()

        if hasattr(self, 'road_analyzer'):
            self.road_analyzer.reset_obstacle_count()

        if not self.is_running:
            self.is_running = True
            self.is_paused = False
            self.road_integrator.reset_roi_smoothing()
            self.speed_tracker.reset()
            self.tracker.reset_bbox_smoothing()
            self.status_bar.config(text=f"Đang phân tích - Chế độ: {self.analysis_mode}")

            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except:
                    pass
            while not self.result_queue.empty():
                try:
                    self.result_queue.get_nowait()
                except:
                    pass

            self.video_thread = threading.Thread(target=self.video_loop, daemon=True)
            self.video_thread.start()
            self.analysis_thread = threading.Thread(target=self.analysis_loop, daemon=True)
            self.analysis_thread.start()
            self.update_ui()

    def video_loop(self):
        if self.cap is None:
            return

        last_frame_time = time.time()

        while self.is_running and self.cap is not None:
            if not self.is_paused:
                current_time = time.time()
                elapsed = current_time - last_frame_time

                if elapsed < self.frame_duration:
                    time.sleep(max(0, self.frame_duration - elapsed))

                ret, frame = self.cap.read()
                if ret:
                    frame = cv2.resize(frame, (640, 480))
                    if self.video_path:
                        self.current_frame += 1
                        if self.total_frames > 0:
                            self.progress_bar['value'] = self.current_frame
                    last_frame_time = time.time()

                    if self.frame_queue.qsize() < 5:
                        self.frame_queue.put(frame)
                else:
                    if self.video_path:
                        self.log("✅ Đã xử lý xong video")
                        self.is_running = False
                    else:
                        self.log("⚠️ Mất kết nối camera")
                    break
            else:
                time.sleep(0.1)

        self.stop_analysis()

    def analysis_loop(self):
        fps_counter = 0
        fps_time = time.time()

        while self.is_running:
            if not self.is_paused and not self.frame_queue.empty():
                frame = self.frame_queue.get()

                try:
                    fh, fw = frame.shape[:2]

                    if self.analysis_mode == "vehicle_only":
                        # ===== CHỈ PHÂN TÍCH PHƯƠNG TIỆN =====
                        roi_coords = [0, fh, 0, fw]
                        road_result = {
                            'roi_coords': roi_coords, 'condition': 'Chỉ phân tích phương tiện',
                            'quality_score': 0, 'pothole_detected': False, 'pothole_count': 0,
                            'crack_detected': False, 'crack_count': 0, 'water_detected': False,
                            'water_count': 0, 'obstacle_detected': False, 'obstacle_count': 0,
                            'total_obstacles': 0, 'edge_density': 0, 'dark_ratio': 0, 'texture_score': 0
                        }
                        frame_with_road = frame.copy()

                        detections, frame_with_detections, vehicle_mask = self.detector.detect(
                            frame_with_road, road_mask=None, return_vehicle_mask=True, roi_coords=roi_coords
                        )

                        # Phát hiện vi phạm
                        violations = self.violation_detector.detect(frame_with_detections, detections, [])

                        for v in violations:
                            if v not in self.violations_log:
                                self.violations_log.append(v)
                                self.log(f"🚨 {v['type']} - {v['description']}")
                                if v['type'] == 'KHONG_DOI_MU':
                                    self.stats['no_helmet'] += 1
                                elif v['type'] == 'VUOT_TOC_DO':
                                    self.stats['speeding'] += 1
                                elif v['type'] == 'SAI_LAN':
                                    self.stats['wrong_lane'] += 1
                                self.stats['violations'] = len(self.violations_log)

                                confident_detections = []
                                bboxes = []
                                confs = []

                                for d in detections:
                                    w = d['bbox'][2] - d['bbox'][0]
                                    h = d['bbox'][3] - d['bbox'][1]

                                    # ĐẠI CHÂM FPS: TUYỆT ĐỐI không ép AI Tracker trích xuất đặc trưng cho cục sạn < 250 pixel
                                    # Giúp lược gánh năng lên Mobilenet -> CPU tăng độ nhạy cực kinh
                                    if w * h > 350 and d['confidence'] >= 0.40:
                                        confident_detections.append(d)
                                        bboxes.append(d['bbox'])
                                        confs.append(d['confidence'])

                                # Truyền bộ nhận nhiện Cực Thu gọn Vào DeepSORT! Tracker múa bay Frame !!!
                                tracks = self.tracker.update(bboxes, frame_with_detections, detections_confs=confs)

                                frame_with_tracks = self.tracker.draw_tracks(frame_with_detections.copy(), tracks)
                                frame_result = frame_with_tracks

                                # Quét rà Check Box Vẫn hoạt Động Hoàn toàn Độc lập Trực tính Của Hệ Không Sai Số:
                                violations = self.violation_detector.detect(frame_with_tracks, confident_detections,
                                                                            tracks)

                                for v in violations:
                                    if v not in self.violations_log:
                                        self.violations_log.append(v)
                                        self.log(f"🚨 {v['type']} - {v['description']}")
                                        if v['type'] == 'KHONG_DOI_MU':
                                            self.stats['no_helmet'] += 1
                                        elif v['type'] == 'VUOT_TOC_DO':
                                            self.stats['speeding'] += 1
                                        elif v['type'] == 'SAI_LAN':
                                            self.stats['wrong_lane'] += 1
                                        self.stats['violations'] = len(self.violations_log)

                                # Logic chống trùng số học Tự Tính
                                for track in tracks:
                                    track_id = track['track_id']
                                    if track_id not in self.counted_track_ids:

                                        best_det = best_detection_for_track(track['bbox'], confident_detections)
                                        if best_det is not None:
                                            class_name = best_det['class_name']

                                            if class_name in ['xe_may', 'xe_oto', 'xe_tai', 'xe_bus', 'xe_dap']:
                                                self.stats['total_vehicles'] += 1

                                            if class_name == 'xe_may':
                                                self.stats['motorcycles'] += 1
                                            elif class_name == 'xe_oto':
                                                self.stats['cars'] += 1
                                            elif class_name == 'xe_tai':
                                                self.stats['trucks'] += 1
                                            elif class_name == 'xe_bus':
                                                self.stats['buses'] += 1
                                            elif class_name == 'xe_dap':
                                                self.stats['bicycles'] += 1
                                            elif class_name == 'nguoi':
                                                self.stats['persons'] += 1

                                            # Cho Tracker nhai kẹo dính vĩnh viễn (ID bị ngậm kén, thoát là đếm xong)
                                            self.counted_track_ids.add(track_id)

                        frame_result = frame_with_detections

                    else:  # analysis_mode == "full" - PHÂN TÍCH TẤT CẢ TRỪ PHƯƠNG TIỆN
                        # ===== PHÂN TÍCH MẶT ĐƯỜNG =====
                        road_result, frame_with_road = self.road_integrator.process(frame)
                        roi_coords = road_result.get('roi_coords', [0, fh, 0, fw])

                        y1, y2, x1, x2 = roi_coords
                        road_region = frame[y1:y2, x1:x2].copy()

                        detailed_road_result = self.road_analyzer.analyze(road_region, None)
                        road_result.update(detailed_road_result)

                        self.stats['obstacles'] = detailed_road_result.get('total_obstacles', 0)

                        # Vẽ kết quả mặt đường
                        frame_result = frame_with_road.copy()
                        for contour in road_result.get('pothole_contours', []):
                            cv2.drawContours(frame_result, [contour], -1, (0, 0, 255), 2)
                        for contour in road_result.get('crack_contours', []):
                            cv2.drawContours(frame_result, [contour], -1, (0, 165, 255), 2)
                        for contour in road_result.get('obstacle_contours', []):
                            cv2.drawContours(frame_result, [contour], -1, (0, 0, 255), 3)

                        # ===== PHÂN TÍCH TƯ THẾ (CHỈ NGƯỜI) - KHÔNG DÙNG DETECTION =====
                        person_detections = []
                        abnormal_poses, frame_result = self.pose_analyzer.analyze(frame_result, person_detections)

                        if abnormal_poses:
                            cv2.putText(frame_result, "⚠️ NGA XE", (50, 80),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                        # Cập nhật UI
                        self.update_road_text(road_result)

                    # Cập nhật stats text
                    self.update_stats_text(self.stats, road_result, [], [])
                    self.update_violation_text(self.violations_log)

                    # Tính FPS
                    fps_counter += 1
                    if time.time() - fps_time >= 1.0:
                        self.fps_display = fps_counter
                        fps_counter = 0
                        fps_time = time.time()

                    # Vẽ thông tin lên frame
                    frame_result = self.draw_info_on_video(frame_result)

                    if self.result_queue.qsize() < 5:
                        self.result_queue.put(frame_result)

                except Exception as e:
                    print(f"Lỗi: {e}")
                    import traceback
                    traceback.print_exc()
                    self.log(f"❌ Lỗi: {str(e)}")

    def draw_info_on_video(self, frame):
        h, w = frame.shape[:2]

        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (280, 95), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

        y = 28
        cv2.putText(frame, f"FPS: {self.fps_display}", (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        y += 22

        if self.analysis_mode == "full":
            cv2.putText(frame, f"Che do: PHAN TICH (Tru phuong tien)", (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1)
            y += 18
            cv2.putText(frame, f"O ga: {self.stats['obstacles']} | Vat can: {self.stats['obstacles']}",
                       (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
        else:
            cv2.putText(frame, f"Che do: CHI PHUONG TIEN", (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 200), 1)
            y += 18
            cv2.putText(frame, f"Xe: {self.stats['total_vehicles']} | Vi pham: {self.stats['violations']}",
                       (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        time_str = datetime.now().strftime("%H:%M:%S")
        (tw, th), _ = cv2.getTextSize(time_str, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (w - tw - 12, h - th - 8), (w - 4, h - 4), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay2, 0.5, frame, 0.5, 0)
        cv2.putText(frame, time_str, (w - tw - 8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        return frame

    def update_ui(self):
        if self.is_running:
            if not self.result_queue.empty():
                frame = self.result_queue.get()
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                pil_image.thumbnail((1024, 768), Image.Resampling.LANCZOS)
                img_tk = ImageTk.PhotoImage(pil_image)

                self.video_label.config(image=img_tk)
                self.video_label.image = img_tk
                self.fps_label.config(text=f"FPS: {self.fps_display}")

                if self.video_path and self.total_frames > 0:
                    current_time = self.current_frame / self.fps_original
                    total_time = self.total_frames / self.fps_original
                    self.time_label.config(text=f"Time: {int(current_time//60):02d}:{int(current_time%60):02d} / {int(total_time//60):02d}:{int(total_time%60):02d}")

            self.root.after(30, self.update_ui)

    def update_stats_text(self, stats, road_result, tracks, violations):
        active_tracks = len(tracks)
        track_ids = [t['track_id'] for t in tracks][:10]

        stats_text = f"""
{'='*70}
THỐNG KÊ - {datetime.now().strftime('%H:%M:%S')}
{'='*70}

🚗 PHƯƠNG TIỆN:
  • Tổng số xe: {stats['total_vehicles']}
  • Xe máy: {stats['motorcycles']}  | Ô tô: {stats['cars']}

🚨 VI PHẠM:
  • Tổng số: {stats['violations']}
  • Không đội mũ: {stats['no_helmet']}
  • Vượt tốc độ: {stats['speeding']}

⚠️ CHƯỚNG NGẠI VẬT: {stats['obstacles']}

🛣️ MẶT ĐƯỜNG:
  • Tình trạng: {road_result['condition']}
  • Ổ gà: {road_result['pothole_count']}

{'='*70}
        """
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(tk.END, stats_text)

    def update_road_text(self, road_result):
        road_text = f"""
{'='*70}
PHÂN TÍCH MẶT ĐƯỜNG - {datetime.now().strftime('%H:%M:%S')}
{'='*70}

📍 Tình trạng: {road_result['condition']}
📍 Điểm chất lượng: {road_result['quality_score']}/100

🔴 Ổ gà: {road_result['pothole_count']}
🟠 Vết nứt: {road_result['crack_count']}
⚠️ Chướng ngại vật: {road_result['obstacle_count']}

{'='*70}
        """
        self.road_text.delete(1.0, tk.END)
        self.road_text.insert(tk.END, road_text)

    def update_violation_text(self, violations_log):
        violation_text = f"""
{'='*70}
DANH SÁCH VI PHẠM - {datetime.now().strftime('%H:%M:%S')}
{'='*70}

🚨 Tổng số vi phạm: {len(violations_log)}
{'─'*70}
"""
        for i, v in enumerate(violations_log[-15:]):
            violation_text += f"\n{i+1}. {v['type']} - {v['description']}"
        violation_text += f"\n{'='*70}"
        self.violation_text.delete(1.0, tk.END)
        self.violation_text.insert(tk.END, violation_text)

    def toggle_pause(self):
        self.is_paused = not self.is_paused
        status = "⏸ Đã tạm dừng" if self.is_paused else "▶ Tiếp tục"
        self.log(status)
        self.status_bar.config(text=status)

    def stop_analysis(self):
        self.is_running = False
        self.is_paused = False
        if self.video_path:
            self.progress_bar['value'] = 0
            self.time_label.config(text="Time: 00:00 / 00:00")
        else:
            self.progress_bar.stop()
        self.status_bar.config(text="Đã dừng")
        self.log("⏹ Đã dừng phân tích")

    def capture_image(self):
        if not self.result_queue.empty():
            frame = self.result_queue.get()
            if not os.path.exists("captures"):
                os.makedirs("captures")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"captures/capture_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            self.log(f"📸 Đã lưu ảnh: {filename}")
            messagebox.showinfo("Thông báo", f"Đã lưu ảnh: {filename}")

    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)

    def on_closing(self):
        self.is_running = False
        if self.cap is not None:
            self.cap.release()
        time.sleep(0.5)
        self.root.destroy()


def main():
    root = tk.Tk()
    app = TrafficAnalysisApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()