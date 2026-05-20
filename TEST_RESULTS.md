# Autonomous Navigation System - Test Results

## Summary
✅ **All 34 tests passing** across 4 modules
✅ **Integration test successful** - Full pipeline runs in simulation mode
✅ **OAK-D Lite ready** - System auto-detects hardware and falls back to simulation

---

## Module Test Results

### 1. AprilTag Detection (test_apriltag.py) - 7/7 ✓
```bash
python test_apriltag.py
```

**Tests:**
- ✓ Initialization - Default and custom parameters
- ✓ Camera Intrinsics - Setting focal length and principal point
- ✓ Synthetic Detection - Tag detection in test images
- ✓ Ground Filtering - Distinguishing ground vs wall tags
- ✓ Pose Estimation - PnP accuracy with known geometry
- ✓ Bearing Calculation - Angle calculations for various positions
- ✓ OAK-D Pipeline - Pipeline structure verification

---

### 2. Ground & Obstacle Detection (test_ground_obstacle.py) - 8/8 ✓
```bash
python test_ground_obstacle.py
```

**Tests:**
- ✓ Ground Detector Init - RANSAC parameter verification
- ✓ Synthetic Ground Detection - Plane detection in point clouds
- ✓ Obstacle Detector Init - Height threshold settings
- ✓ Synthetic Obstacle Detection - Obstacle detection in depth data
- ✓ Path Planner Init - Robot width and clearance settings
- ✓ Cost Map Creation - Navigation cost map generation
- ✓ Path Segment Creation - Path segment validation
- ✓ Pipeline Structure - Integrated pipeline setup

---

### 3. Kalman Filter (test_kalman_filter.py) - 9/9 ✓
```bash
python test_kalman_filter.py
```

**Tests:**
- ✓ Vehicle State Creation - State object creation and conversion
- ✓ EKF Initialization - Filter setup with custom initial state
- ✓ EKF Prediction - Bicycle model prediction
- ✓ EKF Update - Measurement correction
- ✓ Angle Normalization - Angle wrapping to [-π, π]
- ✓ Tag Fusion Init - Measurement fusion initialization
- ✓ Tag Measurement Processing - Tag detection to measurement conversion
- ✓ Multiple Tag Fusion - Combining measurements from multiple tags
- ✓ EKF Reset - Filter reset to initial conditions

---

### 4. MPC Controller (test_mpc_controller.py) - 10/10 ✓
```bash
python test_mpc_controller.py
```

**Tests:**
- ✓ MPC Config - Configuration with constraints
- ✓ Vehicle Dynamics - Bicycle model kinematics
- ✓ Steering Kinematics - Turning radius calculations
- ✓ MPC Controller Init - Optimization controller initialization
- ✓ Dynamics Linearization - Jacobian matrix computation
- ✓ Reference Trajectory - Path reference generation
- ✓ Simple MPC Solve - Control optimization (no obstacles)
- ✓ MPC with Obstacles - Obstacle avoidance
- ✓ Path Following Controller - High-level control interface
- ✓ Fallback Control - PD control when MPC fails

---

## Integration Test

### Run Full Navigation Pipeline
```bash
# Headless mode (no display required)
python main_navigation.py --target 0 --no-display

# With GUI visualization (requires display)
python main_navigation.py --target 0
```

### Expected Output
```
Initializing Autonomous Navigation System...
Initialization complete.
Starting OAK-D device...
Running in simulation mode without OAK-D hardware.
Navigation system started.
[detecting_tags] Pos: (0.00, 0.00), Cmd: a=0.000, s=0.000
[detecting_tags] Pos: (0.00, 0.00), Cmd: a=0.000, s=0.000
...
```

**System Status:**
- ✅ Initializes successfully
- ✅ Runs in "detecting_tags" state
- ✅ Outputs diagnostic information every 100ms
- ✅ Processes frames and generates control commands
- ✅ Auto-detects OAK-D hardware (runs in simulation if not connected)

---

## Quick Start Commands

### Run All Tests
```bash
# Run all test suites
python test_apriltag.py && python test_ground_obstacle.py && \
python test_kalman_filter.py && python test_mpc_controller.py

# Or run individual test suites
python test_apriltag.py          # 7 tests
python test_ground_obstacle.py   # 8 tests
python test_kalman_filter.py     # 9 tests
python test_mpc_controller.py    # 10 tests
```

### Run Integration Test
```bash
# Quick integration test (single frame)
python -c "
from main_navigation import AutonomousNavigator
nav = AutonomousNavigator(target_tag_id=0)
nav.start()
cmd = nav.process_frame()
diag = nav.get_diagnostics()
print(f'State: {diag[\"navigation_state\"]}')
print(f'Command: accel={cmd.acceleration:.3f}, steer={cmd.steering_rate:.3f}')
nav.stop()
"

# Full navigation loop
python main_navigation.py --target 0 --no-display
```

---

## Hardware Requirements

### Without OAK-D Hardware
- All tests pass in simulation mode
- Perfect for development and debugging
- Uses synthetic camera feeds

### With OAK-D Lite
```bash
# Connect OAK-D via USB and run
python main_navigation.py --target 0
```

The system will automatically:
1. Detect connected OAK-D device
2. Stream RGB and depth frames at 30 FPS
3. Perform real-time AprilTag detection
4. Navigate toward detected ground tags
5. Avoid obstacles using MPC control

---

## Troubleshooting

### DepthAI Bus Error
If you see "Bus error" when importing depthai:
- This is normal in containerized environments without USB passthrough
- System automatically falls back to simulation mode
- All logic tests still pass

### OpenCV GUI Error
If you see "The function is not implemented" for cv2.imshow:
- Run with `--no-display` flag
- Install GTK support: `apt-get install libgtk2.0-dev pkg-config`
- Reinstall opencv-python-headless for headless operation

### AprilTag Detection Issues
- Ensure camera intrinsics match your calibration
- Adjust `quad_decimate` parameter for speed/accuracy tradeoff
- Use tag36h11 family for best results

---

## Next Steps

1. **Test with Real Tags**: Print AprilTag markers and place on ground
2. **Connect OAK-D**: Plug in device and run with GUI
3. **Tune Parameters**: Adjust MPC, Kalman filter gains for your vehicle
4. **Deploy**: Run autonomous navigation to target tags

---

**Last Updated:** All tests verified passing
**Total Tests:** 34/34 ✓
