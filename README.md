
## Testing and Debugging

The system includes comprehensive test suites for each module. Run tests to verify functionality at each level.

### Running Individual Test Suites

```bash
# Test AprilTag detection module
python test_apriltag.py

# Test ground plane and obstacle detection
python test_ground_obstacle.py

# Test Kalman filter state estimation
python test_kalman_filter.py

# Test MPC controller
python test_mpc_controller.py
```

### Test Suite Descriptions

#### AprilTag Detection Tests (test_apriltag.py)
1. **Initialization** - Verify detector creation with default/custom parameters
2. **Camera Intrinsics** - Test setting focal length and principal point
3. **Synthetic Detection** - Detect tags in test images (requires test_tag.png)
4. **Ground Filtering** - Filter ground-level vs wall-mounted tags
5. **Pose Estimation** - Verify PnP accuracy with known geometry
6. **Bearing Calculation** - Test angle calculations for various positions
7. **OAK-D Pipeline** - Verify pipeline structure (without hardware)

#### Ground & Obstacle Detection Tests (test_ground_obstacle.py)
1. **Ground Detector Init** - Verify RANSAC parameters
2. **Synthetic Ground Detection** - Detect plane in synthetic point cloud
3. **Obstacle Detector Init** - Verify height threshold settings
4. **Synthetic Obstacle Detection** - Detect obstacles in depth data
5. **Path Planner Init** - Verify robot width and clearance settings
6. **Cost Map Creation** - Generate navigation cost maps
7. **Path Segment Creation** - Create and validate path segments
8. **Pipeline Structure** - Test integrated pipeline setup

#### Kalman Filter Tests (test_kalman_filter.py)
1. **Vehicle State Creation** - Create and convert state objects
2. **EKF Initialization** - Set up filter with custom initial state
3. **EKF Prediction** - Test bicycle model prediction
4. **EKF Update** - Verify measurement correction
5. **Angle Normalization** - Test angle wrapping to [-pi, pi]
6. **Tag Fusion Init** - Initialize measurement fusion
7. **Tag Measurement Processing** - Convert tag detections to measurements
8. **Multiple Tag Fusion** - Combine measurements from multiple tags
9. **EKF Reset** - Reset filter to initial conditions

#### MPC Controller Tests (test_mpc_controller.py)
1. **MPC Config** - Create configuration with constraints
2. **Vehicle Dynamics** - Test bicycle model kinematics
3. **Steering Kinematics** - Verify turning radius calculations
4. **MPC Controller Init** - Initialize optimization controller
5. **Dynamics Linearization** - Compute Jacobian matrices
6. **Reference Trajectory** - Generate path references
7. **Simple MPC Solve** - Optimize control inputs (no obstacles)
8. **MPC with Obstacles** - Test obstacle avoidance
9. **Path Following Controller** - High-level control interface
10. **Fallback Control** - PD control when MPC fails

### Running All Tests

```bash
# Run all test suites sequentially
python test_apriltag.py && python test_ground_obstacle.py && \
python test_kalman_filter.py && python test_mpc_controller.py
```

### Debugging Tips

**AprilTag Detection Issues:**
- If synthetic detection fails, create a test image with an AprilTag
- Check camera intrinsics match your OAK-D calibration
- Adjust quad_decimate for faster/slower detection

**Ground Plane Detection Issues:**
- Increase min_inliers if too many false positives
- Decrease ransac_threshold for stricter plane fitting
- Ensure depth data has sufficient ground points visible

**Kalman Filter Issues:**
- Increase process_noise if filter is too slow to track
- Decrease measurement_noise to trust measurements more
- Check tag map contains correct world positions

**MPC Controller Issues:**
- Reduce horizon if solve time is too long
- Increase obstacle_safety_margin for more conservative navigation
- Check CVXPY solver installation: pip install cvxpy osqp

### Creating Test Data

For testing with real AprilTags, generate tags using the AprilTag tools or download pre-generated tags from the AprilTag GitHub wiki.

## License

MIT License
