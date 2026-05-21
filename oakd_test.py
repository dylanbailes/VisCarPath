import depthai as dai
import cv2
import time
import sys

def print_header(title):
    print("\n" + "="*60)
    print(f"🔍 TEST: {title}")
    print("="*60)

def run_test(test_name, usb_kwargs, setup_func):
    print_header(test_name)
    pipeline = dai.Pipeline()
    try:
        # 1. Setup specific pipeline
        queue_names = setup_func(pipeline)
        
        # 2. Connect device
        print("🔌 Initializing device...")
        with dai.Device(pipeline, **usb_kwargs) as device:
            print(f"✅ CONNECTED | MxId: {device.getMxId()}")
            print(f"   Negotiated USB Speed: {device.getUsbSpeed().name}")
            
            # 3. Create queues
            queues = {name: device.getOutputQueue(name, maxSize=4, blocking=False) for name in queue_names}
            
            # 4. Run loop
            print("📡 Streaming frames... (Press 'q' to skip, or wait 5s)")
            start = time.time()
            frame_count = 0
            window_name = f"Test: {test_name}"
            
            while (time.time() - start) < 5:
                # Try to get first available frame from any active queue
                frame = None
                for q_name, q in queues.items():
                    pkt = q.tryGet()
                    if pkt:
                        frame = pkt.getCvFrame() if q_name != "depth" else pkt.getFrame()
                        break
                
                if frame is not None:
                    frame_count += 1
                    cv2.imshow(window_name, frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    time.sleep(0.01)
                    
            fps = frame_count / max(0.1, time.time() - start)
            print(f"✅ TEST PASSED | ~{fps:.1f} FPS")
            cv2.destroyWindow(window_name)
            return True
            
    except RuntimeError as e:
        print(f"❌ FAILED: {e}")
        return False
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {e}")
        return False
    finally:
        cv2.destroyAllWindows()
        time.sleep(1)

# ================= TEST DEFINITIONS =================
def setup_rgb_only(pipeline):
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setPreviewSize(640, 480)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    out = pipeline.create(dai.node.XLinkOut)
    out.setStreamName("rgb")
    cam.preview.link(out.input)
    return ["rgb"]

def setup_stereo_only(pipeline):
    left = pipeline.create(dai.node.MonoCamera)
    left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    left.setBoardSocket(dai.CameraBoardSocket.LEFT)
    
    right = pipeline.create(dai.node.MonoCamera)
    right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    
    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
    stereo.setOutputSize(640, 360)
    stereo.setRectifyEdgeFillColor(0)
    
    left.out.link(stereo.left)
    right.out.link(stereo.right)
    
    out = pipeline.create(dai.node.XLinkOut)
    out.setStreamName("depth")
    stereo.depth.link(out.input)
    return ["depth"]

def setup_rgb_stereo_aligned(pipeline):
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setPreviewSize(640, 360)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
    cam_rgb.setFps(15)
    
    mono_l = pipeline.create(dai.node.MonoCamera)
    mono_l.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_l.setBoardSocket(dai.CameraBoardSocket.LEFT)
    
    mono_r = pipeline.create(dai.node.MonoCamera)
    mono_r.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_r.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    
    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
    stereo.setOutputSize(640, 360)
    stereo.setRectifyEdgeFillColor(0)
    stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
    stereo.setLeftRightCheck(True)  # REQUIRED for RGB alignment
    stereo.setExtendedDisparity(False)
    stereo.setSubpixel(False)
    
    mono_l.out.link(stereo.left)
    mono_r.out.link(stereo.right)
    
    out_rgb = pipeline.create(dai.node.XLinkOut)
    out_rgb.setStreamName("rgb")
    cam_rgb.preview.link(out_rgb.input)
    
    out_depth = pipeline.create(dai.node.XLinkOut)
    out_depth.setStreamName("depth")
    stereo.depth.link(out_depth.input)
    
    return ["rgb", "depth"]

# ================= RUN ALL TESTS =================
if __name__ == "__main__":
    print("\n🚀 OAK-D Lite USB & Pipeline Diagnostic Suite")
    print("Each test runs for 5 seconds. Press 'q' to skip early.")
    input("Press ENTER to begin...")

    usb2 = {"usb2Mode": True}
    usb3 = {"maxUsbSpeed": dai.UsbSpeed.SUPER}
    
    results = {}

    results["1. RGB Only @ USB 2.0"] = run_test("RGB Only @ USB 2.0", usb2, setup_rgb_only)
    results["2. RGB Only @ USB 3.0"] = run_test("RGB Only @ USB 3.0", usb3, setup_rgb_only)
    
    results["3. Stereo Depth @ USB 2.0"] = run_test("Stereo Only @ USB 2.0", usb2, setup_stereo_only)
    results["4. Stereo Depth @ USB 3.0"] = run_test("Stereo Only @ USB 3.0", usb3, setup_stereo_only)
    
    results["5. RGB+Stereo Aligned @ USB 2.0"] = run_test("RGB+Stereo Aligned @ USB 2.0", usb2, setup_rgb_stereo_aligned)
    results["6. RGB+Stereo Aligned @ USB 3.0"] = run_test("RGB+Stereo Aligned @ USB 3.0", usb3, setup_rgb_stereo_aligned)

    print("\n" + "="*60)
    print("📊 DIAGNOSTIC SUMMARY")
    print("="*60)
    for test, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} | {test}")
    print("="*60)
    print("💡 Next Steps:")
    print("- If ONLY USB 3.0 tests fail: Your Windows USB stack/cable can't handle the VPU power spike. Use usb2Mode=True in production code.")
    print("- If Stereo tests fail but RGB works: Mono camera calibration or alignment config is broken.")
    print("- If ALL tests fail: Driver/cable/port issue. Try different USB port or run on Raspberry Pi.")
    input("\nPress ENTER to exit...")