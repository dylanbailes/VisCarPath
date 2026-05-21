import depthai as dai
import cv2
import numpy as np
import time
from pupil_apriltags import Detector

# ================= CONFIGURATION =================
TAG_FAMILY = "tag36h11"      # Change if using tag16h5, tag25h9, etc.
RESOLUTION = (640, 480)      # Width, Height
QUAD_DECIMATE = 2.0          # 1.0 = full accuracy (slower), 2.0 = faster (recommended), 3.0+ = very fast but misses distant tags
QUAD_SIGMA = 0.0             # Gaussian blur before detection (0.0 = none)
NTHREADS = 1                 # Keep at 1 on Windows/laptops to avoid bus errors
# =================================================

print("Initializing AprilTag Test Pipeline...")

# 1. Initialize AprilTag Detector
tag_detector = Detector(
    families=TAG_FAMILY,
    nthreads=NTHREADS,
    quad_decimate=QUAD_DECIMATE,
    quad_sigma=QUAD_SIGMA
)

# 2. Create DepthAI Pipeline
pipeline = dai.Pipeline()
cam_rgb = pipeline.create(dai.node.ColorCamera)
cam_rgb.setPreviewSize(RESOLUTION[0], RESOLUTION[1])
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_rgb.setInterleaved(False)
cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
cam_rgb.setFps(30)

xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("rgb")
cam_rgb.preview.link(xout_rgb.input)

# 3. Connect to Device (USB 3.0 -> 2.0 fallback)
print("Connecting to OAK-D Lite...")
try:
    device = dai.Device(pipeline, maxUsbSpeed=dai.UsbSpeed.SUPER)
    print("✅ Connected via USB 3.0")
except RuntimeError:
    print("⚠️ USB 3.0 handshake failed. Falling back to USB 2.0...")
    device = dai.Device(pipeline, usb2Mode=True)
    print("✅ Connected via USB 2.0")

q_rgb = device.getOutputQueue(name="rgb", maxSize=3, blocking=False)
print(f"Streaming at {RESOLUTION[0]}x{RESOLUTION[1]} | Decimate: {QUAD_DECIMATE}x | Press 'q' to quit.")

last_fps_time = time.time()
frame_count = 0
fps = 0

try:
    while True:
        det_start = time.time()
        
        # Non-blocking frame fetch
        in_rgb = q_rgb.tryGet()
        if in_rgb is None:
            cv2.waitKey(1)  # Keep OpenCV GUI alive
            continue

        rgb_frame = in_rgb.getCvFrame()
        h, w = rgb_frame.shape[:2]

        # 4. Run AprilTag Detection
        gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
        detections = tag_detector.detect(gray)
        det_time_ms = (time.time() - det_start) * 1000

        # 5. Draw Results
        display = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        
        for det in detections:
            # Draw bounding polygon
            pts = det.corners.astype(np.int32)
            cv2.polylines(display, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            
            # Draw center
            center = tuple(np.mean(det.corners, axis=0).astype(int))
            cv2.circle(display, center, 5, (0, 0, 255), -1)
            
            # Draw tag ID & confidence
            label = f"ID: {det.tag_id} | Conf: {det.decision_margin:.0f}"
            cv2.putText(display, label, (center[0] + 10, center[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw pixel coordinates
            cv2.putText(display, f"({center[0]}, {center[1]})", 
                        (center[0] + 10, center[1] + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # 6. Calculate & Draw FPS
        frame_count += 1
        if time.time() - last_fps_time >= 1.0:
            fps = frame_count
            frame_count = 0
            last_fps_time = time.time()

        info_lines = [
            f"FPS: {fps}",
            f"Det Time: {det_time_ms:.1f}ms",
            f"Tags Found: {len(detections)}",
            f"Resolution: {w}x{h} | Decimate: {QUAD_DECIMATE}x"
        ]
        y = 30
        for line in info_lines:
            cv2.putText(display, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            y += 25

        cv2.imshow("AprilTag Detection Test", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nInterrupted by user")
finally:
    cv2.destroyAllWindows()
    device.close()
    print("✅ Test complete.")