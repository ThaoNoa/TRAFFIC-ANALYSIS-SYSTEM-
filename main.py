"""
Hệ thống phân tích giao thông Lĩnh Nam
Tích hợp: YOLOv8 (phát hiện), DeepSORT (theo dõi),
Pose Analysis, Road Analysis
"""

import tkinter as tk
import cv2
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import queue
import time
from datetime import datetime
import os
import sys
import numpy as np
from modules.utils import draw_info_panel, draw_timestamp, save_frame, iou
# Thêm thư mục hiện tại vào path để import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import modules
from modules.road_analysis import RoadAnalyzer
from modules.road_segmentation_yolov8seg import RoadSegmentorYOLOSeg   # nếu bạn định dùng YOLOv8-Seg
from modules.detection import VehicleDetector
from modules.tracking import SimpleTracker
from modules.road_integrator import RoadIntegrator
from modules.pose_analysis_simple import PoseAnalyzer
from modules.violation_detector import ViolationDetector
from modules.utils import draw_info_panel, draw_timestamp, save_frame
from modules.road_segmentation_yolov8seg import RoadSegmentorYOLOSeg
from modules.ipm_transform import IPMTransformer
from modules.tracking_deepsort import DeepSORTTracker

class TrafficAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hệ thống phân tích giao thông Lĩnh Nam - AI Pro")
        self.root.geometry("1600x900")
        self.road_analyzer = RoadAnalyzer()
        self.use_yolov8_seg = False  # Mặc định dùng OpenCV
        self.segmentor_yolo = None  # Sẽ khởi tạo nếu cần
        self.ipm = None
        self.src_points = None  #[(100, 400), (500, 400), (50, 600), (550, 600)]  # (x,y) trên ảnh gốc
        self.dst_points = None  #[(0, 0), (10, 0), (0, 10), (10, 10)]  # tọa độ thế giới (mét)
        # Biến cho pose analysis theo track
        self.pose_results_per_track = {}  # track_id -> góc nghiêng, confidence

        # Biến cho TensorRT
        self.use_tensorrt = False

        self.track_history = {}
        self.speed_history = {}
        self.pose_results_per_track = {}
        # Khởi tạo các module
        print("=" * 70)
        print("KHỞI TẠO HỆ THỐNG PHÂN TÍCH GIAO THÔNG THÔNG MINH")
        print("=" * 70)

        try:
            print("1. Khởi tạo VehicleDetector (YOLOv8)...")
            self.detector = VehicleDetector()

            print("2. Khởi tạo SimpleTracker...")
            self.tracker = DeepSORTTracker(lambda_motion=0.7, max_age=30)

            print("3. Khởi tạo RoadIntegrator...")
            self.road_integrator = RoadIntegrator()

            print("4. Khởi tạo PoseAnalyzer (Simple)...")
            self.pose_analyzer = PoseAnalyzer()

            print("5. Khởi tạo ViolationDetector...")
            self.violation_detector = ViolationDetector()

            print("✅ Đã khởi tạo xong tất cả modules!")
        except Exception as e:
            print(f"❌ Lỗi khởi tạo module: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Lỗi", f"Không thể khởi tạo module: {e}")
            sys.exit(1)

        # Biến điều khiển
        self.video_path = None
        self.cap = None
        self.is_running = False
        self.is_paused = False
        self.frame_queue = queue.Queue(maxsize=5)
        self.result_queue = queue.Queue(maxsize=5)

        # Thông tin video
        self.total_frames = 0
        self.current_frame = 0
        self.fps_original = 30
        self.frame_duration = 1.0 / 30

        # Tạo giao diện
        self.create_ui()

        # Biến FPS hiển thị
        self.fps_display = 0
        self.frame_count = 0
        self.last_time = time.time()

        # Stats
        self.stats = {
            'total_vehicles': 0,
            'motorcycles': 0,
            'cars': 0,
            'trucks': 0,
            'buses': 0,
            'persons': 0,
            'bicycles': 0,
            'violations': 0
        }

        # Log violations
        self.violations_log = []

        print("=" * 70)
        print("🟢 HỆ THỐNG SẴN SÀNG - Chọn video để bắt đầu")
        print("=" * 70)

    def create_ui(self):
        """Tạo giao diện chính"""
        # Frame chính
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=4)
        main_frame.columnconfigure(1, weight=2)
        main_frame.rowconfigure(0, weight=1)

        # ===== LEFT PANEL - VIDEO DISPLAY =====
        left_frame = ttk.Frame(main_frame)
        left_frame.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        left_frame.columnconfigure(0, weight=1)
        left_frame.rowconfigure(0, weight=1)

        # Video display
        self.video_label = ttk.Label(left_frame, relief=tk.SUNKEN, background='black')
        self.video_label.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Controls
        control_frame = ttk.Frame(left_frame)
        control_frame.grid(row=1, column=0, pady=10)

        ttk.Button(control_frame, text="📁 Mở Video",
                   command=self.open_video).grid(row=0, column=0, padx=5)
        ttk.Button(control_frame, text="▶ Phát",
                   command=self.start_analysis).grid(row=0, column=1, padx=5)
        ttk.Button(control_frame, text="⏸ Tạm dừng",
                   command=self.toggle_pause).grid(row=0, column=2, padx=5)
        ttk.Button(control_frame, text="⏹ Dừng",
                   command=self.stop_analysis).grid(row=0, column=3, padx=5)
        ttk.Button(control_frame, text="📸 Chụp ảnh",
                   command=self.capture_image).grid(row=0, column=4, padx=5)
        # Checkbox chọn phương pháp phân đoạn đường
        self.seg_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(control_frame, text="Dùng YOLOv8-Seg", variable=self.seg_var,
                        command=self.toggle_seg_method).grid(row=0, column=5, padx=5)
        # Checkbox dùng TensorRT
        self.trt_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(control_frame, text="TensorRT", variable=self.trt_var,
                        command=self.toggle_tensorrt).grid(row=0, column=6, padx=5)

        # Info bar
        info_frame = ttk.Frame(left_frame)
        info_frame.grid(row=2, column=0, pady=5, sticky=(tk.W, tk.E))

        self.video_info = ttk.Label(info_frame, text="Chưa chọn video", font=('Arial', 10))
        self.video_info.grid(row=0, column=0, sticky=tk.W)

        self.fps_label = ttk.Label(info_frame, text="FPS: 0", font=('Arial', 10))
        self.fps_label.grid(row=0, column=1, padx=20)

        self.time_label = ttk.Label(info_frame, text="Time: 00:00 / 00:00", font=('Arial', 10))
        self.time_label.grid(row=0, column=2, padx=20)

        # Progress
        self.progress_bar = ttk.Progressbar(left_frame, orient=tk.HORIZONTAL, length=800, mode='determinate')
        self.progress_bar.grid(row=3, column=0, pady=5, sticky=(tk.W, tk.E))

        # ===== RIGHT PANEL - INFORMATION =====
        right_frame = ttk.Frame(main_frame, width=600)
        right_frame.grid(row=0, column=1, padx=5, pady=5, sticky=(tk.N, tk.S, tk.W, tk.E))
        right_frame.grid_propagate(False)
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(0, weight=1)

        # Notebook with tabs
        notebook = ttk.Notebook(right_frame)
        notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # ===== TAB 1: THỐNG KÊ =====
        stats_frame = ttk.Frame(notebook)
        notebook.add(stats_frame, text="📊 Thống kê")
        stats_frame.columnconfigure(0, weight=1)
        stats_frame.rowconfigure(0, weight=1)

        self.stats_text = tk.Text(stats_frame, width=70, height=35, font=('Courier', 9))
        stats_scroll = ttk.Scrollbar(stats_frame, orient="vertical", command=self.stats_text.yview)
        self.stats_text.configure(yscrollcommand=stats_scroll.set)
        self.stats_text.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        stats_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))

        # ===== TAB 2: PHÂN TÍCH MẶT ĐƯỜNG =====
        road_frame = ttk.Frame(notebook)
        notebook.add(road_frame, text="🛣️ Mặt đường")
        road_frame.columnconfigure(0, weight=1)
        road_frame.rowconfigure(0, weight=1)

        self.road_text = tk.Text(road_frame, width=70, height=35, font=('Courier', 9))
        road_scroll = ttk.Scrollbar(road_frame, orient="vertical", command=self.road_text.yview)
        self.road_text.configure(yscrollcommand=road_scroll.set)
        self.road_text.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        road_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))

        # ===== TAB 3: VI PHẠM & CẢNH BÁO =====
        violation_frame = ttk.Frame(notebook)
        notebook.add(violation_frame, text="🚨 Vi phạm")
        violation_frame.columnconfigure(0, weight=1)
        violation_frame.rowconfigure(0, weight=1)

        self.violation_text = tk.Text(violation_frame, width=70, height=35, font=('Courier', 9))
        violation_scroll = ttk.Scrollbar(violation_frame, orient="vertical", command=self.violation_text.yview)
        self.violation_text.configure(yscrollcommand=violation_scroll.set)
        self.violation_text.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        violation_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))

        # ===== TAB 4: LOG HỆ THỐNG =====
        log_frame = ttk.Frame(notebook)
        notebook.add(log_frame, text="📝 Log")
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        self.log_text = tk.Text(log_frame, width=70, height=35, font=('Courier', 9))
        log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        self.log_text.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        log_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))

        # Status bar
        self.status_bar = ttk.Label(self.root, text="Sẵn sàng", relief=tk.SUNKEN)
        self.status_bar.grid(row=1, column=0, sticky=(tk.W, tk.E))

        # Log khởi động
        self.log("🚀 Hệ thống AI phân tích giao thông đã khởi động")
        self.log("📌 Chọn video để bắt đầu phân tích")

    def toggle_seg_method(self):
        self.use_yolov8_seg = self.seg_var.get()
        if self.use_yolov8_seg and self.segmentor_yolo is None:
            # Khởi tạo YOLOv8-seg với fallback là road_integrator hiện tại
            try:
                self.segmentor_yolo = RoadSegmentorYOLOSeg(
                    model_path='yolov8n-seg.pt',  # Đường dẫn model đã fine-tune
                    fallback=self.road_integrator.segmentation  # fallback OpenCV
                )
                self.log("✅ Đã chuyển sang YOLOv8-Seg")
            except Exception as e:
                self.log(f"❌ Không thể khởi tạo YOLOv8-Seg: {e}")
                self.seg_var.set(False)
                self.use_yolov8_seg = False
        else:
            self.log("ℹ️ Đang dùng OpenCV segmentation")

    def toggle_tensorrt(self):
        self.use_tensorrt = self.trt_var.get()
        # Khi thay đổi, cần khởi tạo lại detector với engine mới? Hoặc chỉ ảnh hưởng lần chạy sau.
        # Ở đây ta chỉ log và khi start analysis sẽ dùng giá trị này.
        self.log(f"TensorRT: {'Bật' if self.use_tensorrt else 'Tắt'}")

    def open_video(self):
        """Mở file video"""
        file_path = filedialog.askopenfilename(
            title="Chọn video giao thông",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*")
            ]
        )

        if file_path:
            self.video_path = file_path
            self.log(f"📁 Đã chọn: {os.path.basename(file_path)}")

            # Lấy thông tin video
            cap = cv2.VideoCapture(file_path)
            if cap.isOpened():
                self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.fps_original = cap.get(cv2.CAP_PROP_FPS)
                if self.fps_original <= 0:
                    self.fps_original = 30

                self.frame_duration = 1.0 / self.fps_original

                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                duration = self.total_frames / self.fps_original

                minutes = int(duration // 60)
                seconds = int(duration % 60)

                self.video_info.config(
                    text=f"{os.path.basename(file_path)} | {width}x{height} | {self.fps_original:.1f}fps | {minutes:02d}:{seconds:02d}"
                )
                self.progress_bar['maximum'] = self.total_frames
                self.time_label.config(text=f"Time: 00:00 / {minutes:02d}:{seconds:02d}")
                cap.release()

                self.status_bar.config(text=f"Đã chọn: {os.path.basename(file_path)}")
            else:
                messagebox.showerror("Lỗi", "Không thể mở file video!")

    def start_analysis(self):
        """Bắt đầu phân tích"""
        self.stats = {
            'total_vehicles': 0,
            'motorcycles': 0,
            'cars': 0,
            'trucks': 0,
            'buses': 0,
            'persons': 0,
            'bicycles': 0,
            'violations': 0
        }

        if self.use_tensorrt:
            engine_path = 'yolov8n_fp16.engine'
            if os.path.exists(engine_path):
                self.detector = VehicleDetector(model_path=engine_path, use_tensorrt=True)
                self.log("✅ Đã chuyển sang TensorRT engine")
            else:
                self.log("⚠️ Không tìm thấy TensorRT engine, dùng PyTorch")

        if not self.video_path:
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn video!")
            return

        if not self.is_running:
            self.is_running = True
            self.is_paused = False
            self.status_bar.config(text="Đang phân tích...")
            self.log("▶ Bắt đầu phân tích")

            # Thread đọc video
            self.video_thread = threading.Thread(target=self.video_loop, daemon=True)
            self.video_thread.start()

            # Thread phân tích
            self.analysis_thread = threading.Thread(target=self.analysis_loop, daemon=True)
            self.analysis_thread.start()

            # Thread cập nhật UI
            self.update_ui()

    def video_loop(self):
        """Đọc video với tốc độ real-time"""
        cap = cv2.VideoCapture(self.video_path)
        self.current_frame = 0
        last_frame_time = time.time()

        while self.is_running and self.current_frame < self.total_frames:
            if not self.is_paused:
                current_time = time.time()
                elapsed = current_time - last_frame_time

                # Điều chỉnh tốc độ đọc frame
                if elapsed < self.frame_duration:
                    time.sleep(self.frame_duration - elapsed)

                ret, frame = cap.read()
                if ret:
                    # Resize để xử lý nhanh hơn
                    frame = cv2.resize(frame, (640, 480))
                    self.current_frame += 1
                    last_frame_time = time.time()

                    if self.frame_queue.qsize() < 5:
                        self.frame_queue.put(frame)
                else:
                    break
            else:
                time.sleep(0.1)

        cap.release()
        if self.current_frame >= self.total_frames:
            self.log("✅ Đã xử lý xong video")
            self.is_running = False

    def analysis_loop(self):
        """Phân tích video với tất cả các module AI"""
        fps_counter = 0
        fps_time = time.time()

        while self.is_running:
            if not self.is_paused and not self.frame_queue.empty():
                frame = self.frame_queue.get()

                # === BƯỚC 1: PHÂN TÍCH MẶT ĐƯỜNG (TÁCH RIÊNG) ===
                # Gọi segmentation
                road_mask, sidewalk_mask, roi_coords, seg_info = self.road_integrator.segmentation.extract_road_and_sidewalk(
                    frame)
                y1, y2, x1, x2 = roi_coords
                road_region = self.road_integrator.segmentation.get_road_region(frame, road_mask, roi_coords)

                # Gọi analysis
                road_analysis_result = self.road_integrator.analyzer.analyze(road_region)

                # Kết hợp kết quả
                road_result = {
                    **road_analysis_result,
                    'roi_coords': roi_coords,
                    'road_mask': road_mask,
                    'sidewalk_mask': sidewalk_mask,
                    'road_area': seg_info.get('road_area', 0)
                }

                # --- TỰ VẼ LÊN FRAME (đảm bảo hiển thị) ---
                frame_with_road = frame.copy()

                # Tô màu lòng đường (nếu có)
                if road_mask is not None and np.any(road_mask):
                    overlay = frame_with_road.copy()
                    overlay[road_mask > 0] = (0, 255, 0)  # xanh lá
                    cv2.addWeighted(overlay, 0.15, frame_with_road, 0.85, 0, frame_with_road)

                # Vẽ viền ROI
                cv2.rectangle(frame_with_road, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(frame_with_road, "LONG DUONG CHINH", (x1 + 10, y1 + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Vẽ ổ gà
                for cnt in road_result.get('pothole_contours', []):
                    cnt_abs = cnt.copy()
                    cnt_abs[:, :, 0] += x1
                    cnt_abs[:, :, 1] += y1
                    cv2.drawContours(frame_with_road, [cnt_abs], -1, (0, 0, 255), 2)
                    M = cv2.moments(cnt_abs)
                    if M['m00'] > 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        cv2.putText(frame_with_road, "O GA", (cx - 30, cy - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # Vẽ vết nứt
                for cnt in road_result.get('crack_contours', []):
                    cnt_abs = cnt.copy()
                    cnt_abs[:, :, 0] += x1
                    cnt_abs[:, :, 1] += y1
                    cv2.drawContours(frame_with_road, [cnt_abs], -1, (0, 165, 255), 2)
                else:
                    # Dùng road_integrator cũ
                    road_result, frame_with_road = self.road_integrator.process(frame)
                    print("=== DEBUG ROAD ===")
                    print("road_mask:", road_result.get('road_mask') is not None)
                    if road_result.get('road_mask') is not None:
                        print("road_mask sum:", np.sum(road_result['road_mask'] > 0))
                    print("pothole_count:", road_result.get('pothole_count'))
                    print("crack_count:", road_result.get('crack_count'))
                    print("roi_coords:", road_result.get('roi_coords'))

                    # === BƯỚC 2: PHÁT HIỆN PHƯƠNG TIỆN (YOLOv8) ===
                    # Yêu cầu trả về vehicle mask để loại trừ khỏi phân tích hư hỏng
                    detections, frame_with_detections, vehicle_mask = self.detector.detect(
                        frame_with_road, road_mask, return_vehicle_mask=True
                    )

                    # === BƯỚC 3: THEO DÕI ĐỐI TƯỢNG ===
                    bboxes = [d['bbox'] for d in detections]
                    tracks = self.tracker.update(bboxes)
                    frame_with_tracks = self.tracker.draw_tracks(frame_with_detections.copy(), tracks)

                    for track in tracks:
                        track_id = track['track_id']
                        bbox = track['bbox']
                        frame_idx = self.current_frame
                        if track_id not in self.track_history:
                            self.track_history[track_id] = []
                        self.track_history[track_id].append({'bbox': bbox, 'time': frame_idx})
                        # Giới hạn lịch sử 30 frame
                        if len(self.track_history[track_id]) > 30:
                            self.track_history[track_id].pop(0)

                        # Tính vận tốc nếu có IPM
                        if self.ipm is not None:
                            speed = self.ipm.compute_speed(self.track_history[track_id], self.fps_original)
                            if speed is not None:
                                if track_id not in self.speed_history:
                                    self.speed_history[track_id] = []
                                self.speed_history[track_id].append(speed)
                                if len(self.speed_history[track_id]) > 30:
                                    self.speed_history[track_id].pop(0)
                                track['speed'] = speed
                                track['speed_history'] = self.speed_history[track_id]

                    # === BƯỚC 4: PHÂN TÍCH TƯ THẾ ===
                    person_detections = [d for d in detections if d['class_name'] == 'nguoi']
                    abnormal_poses = []
                    if person_detections:
                        abnormal_poses, frame_with_tracks = self.pose_analyzer.analyze(frame_with_tracks, person_detections)
                        self.pose_results_per_track.clear()
                        for pose in abnormal_poses:
                            # Tìm track phù hợp nhất dựa trên IoU
                            best_iou = 0
                            best_track = None
                            for track in tracks:
                                iou_val = iou(pose['bbox'], track['bbox'])
                                if iou_val > best_iou:
                                    best_iou = iou_val
                                    best_track = track
                            if best_track and best_iou > 0.5:
                                self.pose_results_per_track[best_track['track_id']] = pose

                    # === BƯỚC 5: PHÂN TÍCH MẶT ĐƯỜNG CHI TIẾT (loại trừ xe) ===
                    # Lấy vùng lòng đường để phân tích
                    y1, y2, x1, x2 = road_result.get('roi_coords', [0, frame.shape[0], 0, frame.shape[1]])
                    road_region = frame[y1:y2, x1:x2].copy()

                    # Cắt vehicle mask theo vùng lòng đường
                    if vehicle_mask is not None and vehicle_mask.size > 0:
                        vehicle_mask_crop = vehicle_mask[y1:y2, x1:x2]
                    else:
                        vehicle_mask_crop = None

                    # Phân tích mặt đường, loại trừ vùng có xe
                    from modules.road_analysis import RoadAnalyzer
                    road_analyzer = RoadAnalyzer()
                    detailed_road_result = road_analyzer.analyze(road_region, vehicle_mask_crop)

                    # Cập nhật kết quả
                    road_result.update(detailed_road_result)

                    # === BƯỚC 6: PHÁT HIỆN VI PHẠM ===
                    violations = self.violation_detector.detect(
                        frame_with_tracks,
                        detections,
                        tracks,
                        ipm=self.ipm,
                        pose_results=self.pose_results_per_track
                    )

                    # Log violations mới
                    for v in violations:
                        if v not in self.violations_log:
                            self.violations_log.append(v)
                            self.log(f"🚨 {v['type']} - {v['description']}")

                    # === BƯỚC 7: CẬP NHẬT STATS ===
                    frame_stats = self.detector.get_stats()
                    for key in frame_stats:
                        if key in self.stats:
                            self.stats[key] += frame_stats[key]
                            self.stats['violations'] = len(self.violations_log)

                    # === BƯỚC 8: CẬP NHẬT UI TEXT ===
                    self.update_stats_text(self.stats, road_result, tracks, violations)
                    self.update_road_text(road_result)
                    self.update_violation_text(violations)

                    # Tính FPS xử lý
                    fps_counter += 1
                    if time.time() - fps_time >= 1.0:
                        self.fps_display = fps_counter
                        fps_counter = 0
                        fps_time = time.time()

                    # Thêm thông tin lên frame
                    frame_with_tracks = draw_info_panel(frame_with_tracks, self.stats, self.fps_display)
                    frame_with_tracks = draw_timestamp(frame_with_tracks)

                    # Vẽ cảnh báo nếu có
                    if abnormal_poses:
                        cv2.putText(frame_with_tracks, "⚠️ PHAT HIEN NGA XE",
                                   (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    if violations:
                        cv2.putText(frame_with_tracks, f"🚨 VI PHAM: {len(violations)}",
                                   (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                    if road_result.get('pothole_detected', False):
                        cv2.putText(frame_with_tracks, f"🕳️ O GA: {road_result['pothole_count']}",
                                   (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                    if self.result_queue.qsize() < 5:
                        self.result_queue.put(frame_with_tracks)

                #except Exception as e:
                  #  print(f"Lỗi trong analysis_loop: {e}")
                    #import traceback
                  #  traceback.print_exc()
                   # self.log(f"❌ Lỗi: {str(e)}")

    def update_ui(self):
        """Cập nhật giao diện"""
        if self.is_running:
            if not self.result_queue.empty():
                frame = self.result_queue.get()

                # Hiển thị frame
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                pil_image.thumbnail((1024, 768), Image.Resampling.LANCZOS)
                img_tk = ImageTk.PhotoImage(pil_image)

                self.video_label.config(image=img_tk)
                self.video_label.image = img_tk

                # Cập nhật thông tin
                self.fps_label.config(text=f"FPS: {self.fps_display}")

                current_time = self.current_frame / self.fps_original if self.fps_original > 0 else 0
                total_time = self.total_frames / self.fps_original if self.fps_original > 0 else 0

                current_min = int(current_time // 60)
                current_sec = int(current_time % 60)
                total_min = int(total_time // 60)
                total_sec = int(total_time % 60)

                self.time_label.config(
                    text=f"Time: {current_min:02d}:{current_sec:02d} / {total_min:02d}:{total_sec:02d}"
                )

                self.progress_bar['value'] = self.current_frame

            self.root.after(30, self.update_ui)

    def update_stats_text(self, stats, road_result, tracks, violations):
        """Cập nhật text thống kê"""
        y1, y2, x1, x2 = road_result.get('roi_coords', [0, 0, 0, 0])

        # Thống kê tracks
        active_tracks = len(tracks)
        track_ids = [t['track_id'] for t in tracks][:10]

        stats_text = f"""
{'='*70}
THỐNG KÊ GIAO THÔNG - {datetime.now().strftime('%H:%M:%S')}
{'='*70}

🚗 PHƯƠNG TIỆN (ĐỘNG):
  • Tổng số xe    : {stats['total_vehicles']}
  • Xe máy        : {stats['motorcycles']}
  • Ô tô          : {stats['cars']}
  • Xe tải        : {stats['trucks']}
  • Xe buýt       : {stats['buses']}
  • Xe đạp        : {stats['bicycles']}
  • Người         : {stats['persons']}

🎯 THEO DÕI:
  • Đối tượng     : {active_tracks}
  • Track IDs     : {track_ids}

🚨 VI PHẠM:
  • Tổng số       : {stats['violations']}
  • Vi phạm mới   : {len(violations)}

🛣️ MẶT ĐƯỜNG (TĨNH):
  • Tình trạng     : {road_result['condition']}
  • Điểm chất lượng: {road_result['quality_score']}/100
  • Ổ gà           : {road_result['pothole_count']} cái {'⚠️' if road_result['pothole_detected'] else '✅'}
  • Vết nứt        : {road_result['crack_count']} vết {'⚠️' if road_result['crack_detected'] else '✅'}
  • Vũng nước      : {road_result.get('water_count', 0)} cái {'⚠️' if road_result.get('water_detected', False) else '✅'}

📐 THÔNG SỐ KỸ THUẬT:
  • Mật độ cạnh    : {road_result['edge_density']:.3f}
  • Tỷ lệ vùng tối : {road_result['dark_ratio']:.3f}
  • Độ nhám texture: {road_result['texture_score']:.2f}
  • Diện tích đường: {road_result.get('road_area', 0)} pixels

🎯 VÙNG PHÂN TÍCH:
  • Tọa độ ROI      : ({x1},{y1}) - ({x2},{y2})

⏱️ HỆ THỐNG:
  • Thời gian       : {self.current_frame / self.fps_original:.1f}s / {self.total_frames / self.fps_original:.1f}s
  • FPS xử lý       : {self.fps_display}
  • Trạng thái      : {"Đang chạy" if self.is_running else "Dừng"}
  • Tạm dừng        : {"Có" if self.is_paused else "Không"}
{'='*70}
  • Phương pháp đường: {'YOLOv8-Seg' if self.use_yolov8_seg else 'OpenCV'}
  • TensorRT: {'Bật' if self.use_tensorrt else 'Tắt'}
  • IPM: {'Có' if self.ipm else 'Chưa'}
"""
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(tk.END, stats_text)

    def update_road_text(self, road_result):
        """Cập nhật text chi tiết mặt đường"""
        road_text = f"""
{'='*70}
PHÂN TÍCH MẶT ĐƯỜNG - {datetime.now().strftime('%H:%M:%S')}
{'='*70}

📍 KẾT QUẢ PHÂN TÍCH:
  • Tình trạng: {road_result['condition']}
  • Điểm chất lượng: {road_result['quality_score']}/100

🔴 Ổ GÀ (TĨNH):
  • Số lượng: {road_result['pothole_count']}
  • Phát hiện: {'CÓ' if road_result['pothole_detected'] else 'KHÔNG'}

🟠 VẾT NỨT (TĨNH):
  • Số lượng: {road_result['crack_count']}
  • Phát hiện: {'CÓ' if road_result['crack_detected'] else 'KHÔNG'}

💧 VŨNG NƯỚC (TĨNH):
  • Số lượng: {road_result.get('water_count', 0)}
  • Phát hiện: {'CÓ' if road_result.get('water_detected', False) else 'KHÔNG'}

📊 CHỈ SỐ KỸ THUẬT:
  • Mật độ cạnh: {road_result['edge_density']:.4f}
  • Tỷ lệ vùng tối: {road_result['dark_ratio']:.4f}
  • Độ nhám texture: {road_result['texture_score']:.2f}

📍 VÙNG PHÂN TÍCH:
  • Diện tích: {road_result.get('road_area', 0)} pixels
  • Đã loại trừ phương tiện khỏi phân tích hư hỏng

{'='*70}
        """
        self.road_text.delete(1.0, tk.END)
        self.road_text.insert(tk.END, road_text)

    def update_violation_text(self, violations):
        """Cập nhật text vi phạm"""
        violation_text = f"""
{'='*70}
DANH SÁCH VI PHẠM - {datetime.now().strftime('%H:%M:%S')}
{'='*70}

🚨 Tổng số vi phạm: {len(self.violations_log)}

{'─'*70}
"""
        # 10 vi phạm gần nhất
        for i, v in enumerate(self.violations_log[-10:]):
            violation_text += f"""
{i+1}. [{v['time']}] {v['type']}
     • {v['description']}
     • ID: {v.get('track_id', 'N/A')}
"""

        violation_text += f"""
{'─'*70}

⚠️ VI PHẠM MỚI NHẤT ({len(violations)}):
"""
        for v in violations:
            violation_text += f"  • {v['type']}: {v['description']}\n"

        violation_text += f"""
{'='*70}
        """
        self.violation_text.delete(1.0, tk.END)
        self.violation_text.insert(tk.END, violation_text)

    def toggle_pause(self):
        """Tạm dừng / tiếp tục"""
        self.is_paused = not self.is_paused
        status = "⏸ Đã tạm dừng" if self.is_paused else "▶ Tiếp tục"
        self.log(status)
        self.status_bar.config(text=status)

    def stop_analysis(self):
        """Dừng phân tích"""
        self.is_running = False
        self.is_paused = False
        self.progress_bar['value'] = 0
        self.time_label.config(text="Time: 00:00 / 00:00")
        self.status_bar.config(text="Đã dừng")
        self.log("⏹ Đã dừng phân tích")

    def capture_image(self):
        """Chụp ảnh"""
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
        """Ghi log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"
        self.log_text.insert(tk.END, log_message)
        self.log_text.see(tk.END)

    def on_closing(self):
        """Xử lý khi đóng cửa sổ"""
        self.is_running = False
        time.sleep(0.5)
        self.log("👋 Đã đóng ứng dụng")
        self.root.destroy()


def main():
    """Hàm chính để chạy ứng dụng"""
    root = tk.Tk()
    app = TrafficAnalysisApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()