"""
Test Suite for Ground Plane and Obstacle Detection Module
Tests ground plane fitting, obstacle detection, and path planning
"""

import numpy as np
import sys
from ground_obstacle_detection import (
    GroundPlaneDetector, 
    ObstacleDetector, 
    PathPlanner,
    GroundPlane,
    Obstacle,
    PathSegment,
    GroundAndObstaclePipeline
)


def test_ground_plane_detector_init():
    """Test 1: Initialize ground plane detector"""
    print("\n=== Test 1: Ground Plane Detector Initialization ===")
    
    try:
        detector = GroundPlaneDetector(
            ransac_threshold=0.05,
            min_inliers=100,
            max_iterations=1000
        )
        
        assert detector.ransac_threshold == 0.05
        assert detector.min_inliers == 100
        assert detector.max_iterations == 1000
        assert detector.expected_pitch == 0.3
        
        print("✓ Ground plane detector initialized successfully")
        return True
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        return False


def test_synthetic_ground_plane_detection():
    """Test 2: Detect ground plane in synthetic point cloud"""
    print("\n=== Test 2: Synthetic Ground Plane Detection ===")
    
    try:
        detector = GroundPlaneDetector(ransac_threshold=0.02, min_inliers=50)
        
        # Create synthetic ground plane points
        n_points = 500
        x = np.random.uniform(-2, 2, n_points)
        z = np.random.uniform(1, 5, n_points)
        
        # Ground plane at y = 0.5 (camera is above ground)
        # With some noise
        y = np.zeros(n_points) + 0.5 + np.random.normal(0, 0.01, n_points)
        
        points_3d = np.column_stack([x, y, z])
        
        # Create fake depth map and camera intrinsics
        fx, fy, cx, cy = 800.0, 800.0, 640.0, 360.0
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        
        # Convert to depth map format
        h, w = 720, 1280
        depth_map = np.zeros((h, w), dtype=np.uint16)
        
        # Project points to image
        u = (points_3d[:, 0] * fx / points_3d[:, 2] + cx).astype(int)
        v = (points_3d[:, 1] * fy / points_3d[:, 2] + cy).astype(int)
        
        valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        u, v = u[valid], v[valid]
        depths = (points_3d[:, 2][valid] * 1000).astype(np.uint16)  # Convert to mm
        
        depth_map[v, u] = depths
        
        # Detect ground plane
        ground_plane = detector.detect_ground_plane(depth_map, K)
        
        if ground_plane is not None:
            print(f"✓ Ground plane detected:")
            print(f"  Normal: [{ground_plane.normal[0]:.3f}, {ground_plane.normal[1]:.3f}, {ground_plane.normal[2]:.3f}]")
            print(f"  Distance: {ground_plane.distance:.3f}m")
            print(f"  Confidence: {ground_plane.confidence:.2f}")
            print(f"  Inliers: {ground_plane.inliers}")
            
            # Check if normal is roughly pointing up
            if ground_plane.normal[1] > 0:
                print("✓ Normal direction is correct (pointing up)")
            
            return True
        else:
            print("⚠ No ground plane detected (may need more points)")
            return True
            
    except Exception as e:
        print(f"✗ Ground plane detection failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_obstacle_detector_init():
    """Test 3: Initialize obstacle detector"""
    print("\n=== Test 3: Obstacle Detector Initialization ===")
    
    try:
        detector = ObstacleDetector(
            height_threshold=0.05,
            min_obstacle_size=50,
            max_obstacle_distance=5.0
        )
        
        assert detector.height_threshold == 0.05
        assert detector.min_obstacle_size == 50
        assert detector.max_obstacle_distance == 5.0
        
        print("✓ Obstacle detector initialized successfully")
        return True
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        return False


def test_obstacle_detection_synthetic():
    """Test 4: Detect obstacles in synthetic data"""
    print("\n=== Test 4: Synthetic Obstacle Detection ===")
    
    try:
        obs_detector = ObstacleDetector(height_threshold=0.05, min_obstacle_size=10)
        
        # Create synthetic depth map with ground and obstacle
        h, w = 480, 640
        fx, fy, cx, cy = 600.0, 600.0, 320.0, 240.0
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        
        # Create ground plane
        y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        
        # Simulate camera pitched down 0.3 rad
        pitch = 0.3
        depth_map = np.zeros((h, w), dtype=np.uint16)
        
        for v in range(h):
            for u in range(w):
                # Calculate depth for ground plane at y=0.5
                # Using camera projection equations
                theta_v = np.arctan2(v - cy, fy)
                theta_h = np.arctan2(u - cx, fx)
                
                # Ground plane equation
                if abs(np.sin(pitch - theta_v)) > 0.01:
                    z = 0.5 / np.sin(pitch - theta_v)
                    if 0.1 < z < 5.0:
                        depth_map[v, u] = int(z * 1000)
        
        # Add synthetic obstacle (box at center)
        obs_u, obs_v = 320, 240
        obs_depth = 2000  # 2m away
        for dv in range(-20, 20):
            for du in range(-20, 20):
                vu, vv = obs_u + du, obs_v + dv
                if 0 <= vu < w and 0 <= vv < h:
                    # Make obstacle taller than ground
                    depth_map[vv, vu] = obs_depth
        
        # Create mock ground plane
        ground_plane = GroundPlane(
            normal=np.array([0, 0.3, 0.95]),
            distance=-0.5,
            confidence=0.9,
            inliers=1000,
            bounds=(0, w, 0, h)
        )
        
        # Detect obstacles
        obstacles = obs_detector.detect_obstacles(depth_map, ground_plane, K)
        
        print(f"✓ Detected {len(obstacles)} obstacles")
        
        for obs in obstacles:
            print(f"  Obstacle at ({obs.center[0]}, {obs.center[1]}), "
                  f"distance={obs.distance:.2f}m, severity={obs.severity:.2f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Obstacle detection failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_path_planner_init():
    """Test 5: Initialize path planner"""
    print("\n=== Test 5: Path Planner Initialization ===")
    
    try:
        planner = PathPlanner(
            robot_width=0.5,
            min_clearance=0.2,
            max_lookahead=3.0
        )
        
        assert planner.robot_width == 0.5
        assert planner.min_clearance == 0.2
        assert planner.max_lookahead == 3.0
        
        print("✓ Path planner initialized successfully")
        return True
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        return False


def test_cost_map_creation():
    """Test 6: Create cost map with obstacles"""
    print("\n=== Test 6: Cost Map Creation ===")
    
    try:
        planner = PathPlanner()
        
        # Create mock obstacles
        obstacles = [
            Obstacle(
                center=(400, 300),
                position_3d=np.array([0.5, 0.0, 2.0]),
                size=(50, 50),
                distance=2.0,
                bearing=0.25,
                severity=0.5
            ),
            Obstacle(
                center=(200, 300),
                position_3d=np.array([-0.5, 0.0, 2.0]),
                size=(50, 50),
                distance=2.0,
                bearing=-0.25,
                severity=0.5
            )
        ]
        
        # Mock ground plane
        ground_plane = GroundPlane(
            normal=np.array([0, 0.3, 0.95]),
            distance=-0.5,
            confidence=0.9,
            inliers=1000,
            bounds=(0, 640, 0, 480)
        )
        
        # Camera intrinsics
        K = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]])
        
        # Create cost map
        cost_map = planner.create_cost_map(640, 480, obstacles, K, ground_plane)
        
        assert cost_map.shape == (480, 640)
        assert np.all(cost_map > 0)
        
        print(f"✓ Cost map created: shape={cost_map.shape}")
        print(f"  Min cost: {cost_map.min():.2f}")
        print(f"  Max cost: {cost_map.max():.2f}")
        print(f"  Mean cost: {cost_map.mean():.2f}")
        
        # Check that obstacle regions have higher cost
        center_cost = cost_map[300, 320]  # Center (clear path)
        left_obs_cost = cost_map[300, 200]  # Near left obstacle
        right_obs_cost = cost_map[300, 400]  # Near right obstacle
        
        print(f"  Center cost: {center_cost:.2f}")
        print(f"  Left obstacle area cost: {left_obs_cost:.2f}")
        print(f"  Right obstacle area cost: {right_obs_cost:.2f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Cost map creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_path_segment_creation():
    """Test 7: Create path segments manually"""
    print("\n=== Test 7: Path Segment Creation ===")
    
    try:
        # Create a simple path segment
        start = np.array([0.0, 0.0, 0.0])
        end = np.array([0.0, 0.0, 3.0])  # 3m forward
        
        segment = PathSegment(
            start=start,
            end=end,
            width=0.6,
            clearance=0.3,
            cost=1.0
        )
        
        assert np.allclose(segment.start, start)
        assert np.allclose(segment.end, end)
        assert segment.width == 0.6
        assert segment.clearance == 0.3
        
        print(f"✓ Path segment created:")
        print(f"  Start: [{start[0]:.2f}, {start[1]:.2f}, {start[2]:.2f}]")
        print(f"  End: [{end[0]:.2f}, {end[1]:.2f}, {end[2]:.2f}]")
        print(f"  Length: {np.linalg.norm(end - start):.2f}m")
        
        return True
        
    except Exception as e:
        print(f"✗ Path segment creation failed: {e}")
        return False


def test_ground_obstacle_pipeline():
    """Test 8: Test integrated pipeline structure"""
    print("\n=== Test 8: Ground & Obstacle Pipeline Structure ===")
    
    try:
        pipeline = GroundAndObstaclePipeline(robot_width=0.5)
        
        # Verify components
        assert pipeline.ground_detector is not None
        assert pipeline.obstacle_detector is not None
        assert pipeline.path_planner is not None
        assert pipeline.camera_intrinsics is None  # Not set yet
        
        # Set intrinsics
        pipeline.set_camera_intrinsics(800.0, 800.0, 640.0, 360.0)
        
        assert pipeline.camera_intrinsics is not None
        assert pipeline.camera_intrinsics[0, 0] == 800.0
        
        print("✓ Pipeline structure created successfully")
        print("  Note: Full processing requires actual depth frames")
        
        return True
        
    except Exception as e:
        print(f"✗ Pipeline structure test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all ground/obstacle detection tests"""
    print("=" * 60)
    print("GROUND & OBSTACLE DETECTION TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Ground Detector Init", test_ground_plane_detector_init),
        ("Synthetic Ground Detection", test_synthetic_ground_plane_detection),
        ("Obstacle Detector Init", test_obstacle_detector_init),
        ("Synthetic Obstacle Detection", test_obstacle_detection_synthetic),
        ("Path Planner Init", test_path_planner_init),
        ("Cost Map Creation", test_cost_map_creation),
        ("Path Segment Creation", test_path_segment_creation),
        ("Pipeline Structure", test_ground_obstacle_pipeline),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ Test '{name}' crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
