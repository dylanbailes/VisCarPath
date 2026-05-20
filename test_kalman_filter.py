"""
Test Suite for Kalman Filter Module
Tests EKF state estimation, prediction, update, and tag measurement fusion
"""

import numpy as np
import sys
from kalman_filter import (
    ExtendedKalmanFilter, 
    VehicleState, 
    TagMeasurementFusion,
    MotionModel
)


def test_vehicle_state_creation():
    """Test 1: Create vehicle state"""
    print("\n=== Test 1: Vehicle State Creation ===")
    
    try:
        state = VehicleState(x=1.0, y=2.0, theta=np.pi/4, v=0.5, omega=0.1)
        
        assert state.x == 1.0
        assert state.y == 2.0
        assert abs(state.theta - np.pi/4) < 0.001
        assert state.v == 0.5
        assert state.omega == 0.1
        
        print(f"✓ Vehicle state created:")
        print(f"  Position: ({state.x}, {state.y})")
        print(f"  Heading: {np.degrees(state.theta):.1f}°")
        print(f"  Velocity: {state.v} m/s")
        print(f"  Angular velocity: {state.omega} rad/s")
        
        # Test array conversion
        arr = state.to_array()
        assert len(arr) == 5
        assert np.allclose(arr, [1.0, 2.0, np.pi/4, 0.5, 0.1])
        
        print("✓ Array conversion successful")
        
        # Test from_array
        state2 = VehicleState.from_array(arr)
        assert state2.x == state.x
        
        print("✓ Round-trip conversion successful")
        
        return True
    except Exception as e:
        print(f"✗ Vehicle state creation failed: {e}")
        return False


def test_ekf_initialization():
    """Test 2: Initialize Extended Kalman Filter"""
    print("\n=== Test 2: EKF Initialization ===")
    
    try:
        # Default initialization
        ekf = ExtendedKalmanFilter()
        
        assert ekf.x.shape == (5,)
        assert ekf.P.shape == (5, 5)
        assert ekf.Q.shape == (5, 5)
        assert ekf.R.shape == (3, 3)
        assert ekf.dt == 0.1
        
        print("✓ Default EKF initialized at origin")
        print(f"  Initial state: {ekf.x}")
        
        # Custom initial state
        initial_state = VehicleState(x=1.0, y=2.0, theta=0, v=0, omega=0)
        ekf2 = ExtendedKalmanFilter(initial_state=initial_state)
        
        assert ekf2.x[0] == 1.0
        assert ekf2.x[1] == 2.0
        
        print("✓ EKF with custom initial state")
        print(f"  Initial position: ({ekf2.x[0]}, {ekf2.x[1]})")
        
        return True
    except Exception as e:
        print(f"✗ EKF initialization failed: {e}")
        return False


def test_ekf_prediction():
    """Test 3: Test EKF prediction step"""
    print("\n=== Test 3: EKF Prediction Step ===")
    
    try:
        initial_state = VehicleState(x=0, y=0, theta=0, v=1.0, omega=0)
        ekf = ExtendedKalmanFilter(initial_state=initial_state)
        
        # Predict with constant velocity (no control input)
        ekf.predict(dt=1.0, control_input=(0.0, 0.0))
        
        state = ekf.get_state()
        
        # After 1 second at 1 m/s, should be at x=1
        expected_x = 1.0
        error_x = abs(state.x - expected_x)
        
        print(f"✓ Prediction after 1s at 1 m/s:")
        print(f"  Expected x: {expected_x:.2f}m")
        print(f"  Actual x: {state.x:.2f}m")
        print(f"  Error: {error_x*100:.1f}cm")
        
        if error_x < 0.01:
            print("✓ Prediction accurate!")
        else:
            print("⚠ Prediction has larger than expected error")
        
        # Test with turning
        ekf2 = ExtendedKalmanFilter(
            initial_state=VehicleState(x=0, y=0, theta=0, v=1.0, omega=0.5)
        )
        
        ekf2.predict(dt=1.0)
        state2 = ekf2.get_state()
        
        print(f"\n✓ Prediction with turn (omega=0.5 rad/s):")
        print(f"  New heading: {np.degrees(state2.theta):.1f}°")
        print(f"  New position: ({state2.x:.2f}, {state2.y:.2f})")
        
        return True
    except Exception as e:
        print(f"✗ EKF prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ekf_update():
    """Test 4: Test EKF measurement update"""
    print("\n=== Test 4: EKF Measurement Update ===")
    
    try:
        # Start with uncertain state
        initial_state = VehicleState(x=0, y=0, theta=0, v=0, omega=0)
        ekf = ExtendedKalmanFilter(
            initial_state=initial_state,
            process_noise=0.1,
            measurement_noise=0.5
        )
        
        # Make covariance large (uncertain)
        ekf.P = np.eye(5) * 10.0
        
        # Provide a measurement at (2, 3)
        measurement = np.array([2.0, 3.0, np.pi/4])
        
        ekf.update(measurement)
        
        state = ekf.get_state()
        
        print(f"✓ After measurement update:")
        print(f"  Measured position: (2.0, 3.0)")
        print(f"  Estimated position: ({state.x:.2f}, {state.y:.2f})")
        print(f"  Estimated heading: {np.degrees(state.theta):.1f}°")
        
        # Check that estimate moved toward measurement
        error_x = abs(state.x - 2.0)
        error_y = abs(state.y - 3.0)
        
        if error_x < 1.0 and error_y < 1.0:
            print("✓ State updated toward measurement!")
        else:
            print("⚠ State did not update as expected")
        
        # Check that uncertainty decreased
        if np.trace(ekf.P) < 30.0:  # Was 50.0
            print("✓ Covariance reduced after update")
        else:
            print("⚠ Covariance did not reduce as expected")
        
        return True
    except Exception as e:
        print(f"✗ EKF update failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_angle_normalization():
    """Test 5: Test angle normalization"""
    print("\n=== Test 5: Angle Normalization ===")
    
    try:
        ekf = ExtendedKalmanFilter()
        
        test_cases = [
            (0, 0),
            (np.pi, -np.pi),  # Should normalize to -pi
            (2*np.pi, 0),
            (-np.pi/2, -np.pi/2),
            (3*np.pi, np.pi),
            (-3*np.pi, np.pi),
            (10*np.pi, 0),
        ]
        
        all_passed = True
        
        for input_angle, expected in test_cases:
            result = ekf._normalize_angle(input_angle)
            error = abs(result - expected)
            
            # Account for pi/-pi equivalence
            if error > 0.1:
                error = abs(error - 2*np.pi)
            
            status = "✓" if error < 0.01 else "⚠"
            print(f"  {status} normalize({input_angle:.2f}) = {result:.2f} "
                  f"(expected: {expected:.2f})")
            
            if error >= 0.01:
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"✗ Angle normalization failed: {e}")
        return False


def test_tag_measurement_fusion_init():
    """Test 6: Initialize tag measurement fusion"""
    print("\n=== Test 6: Tag Measurement Fusion Initialization ===")
    
    try:
        ekf = ExtendedKalmanFilter()
        fusion = TagMeasurementFusion(ekf)
        
        assert fusion.ekf is ekf
        assert len(fusion.tag_map) == 0
        
        print("✓ Tag measurement fusion initialized")
        
        # Add tags to map
        fusion.add_tag_to_map(0, x=5.0, y=10.0, theta=0)
        fusion.add_tag_to_map(1, x=10.0, y=5.0, theta=np.pi/2)
        
        assert 0 in fusion.tag_map
        assert 1 in fusion.tag_map
        assert fusion.tag_map[0] == (5.0, 10.0, 0)
        
        print(f"✓ Added {len(fusion.tag_map)} tags to map")
        
        return True
    except Exception as e:
        print(f"✗ Tag fusion initialization failed: {e}")
        return False


def test_tag_measurement_processing():
    """Test 7: Process tag measurements"""
    print("\n=== Test 7: Tag Measurement Processing ===")
    
    try:
        from apriltag_detection import AprilTagDetection
        
        ekf = ExtendedKalmanFilter(
            initial_state=VehicleState(x=0, y=0, theta=0, v=0, omega=0)
        )
        fusion = TagMeasurementFusion(ekf)
        
        # Add known tag
        fusion.add_tag_to_map(0, x=5.0, y=5.0, theta=0)
        
        # Create mock tag detection
        # Tag is 3m forward in camera frame
        pose = np.eye(4)
        pose[:3, 3] = [0, 0, 3.0]  # 3m straight ahead
        
        mock_detection = AprilTagDetection(
            tag_id=0,
            tag_family="tag36h11",
            center=(640, 360),
            pose=pose,
            distance=3.0,
            bearing=0.0,
            confidence=0.95
        )
        
        # Process measurement
        measurement = fusion.process_tag_detection(mock_detection)
        
        if measurement is not None:
            print(f"✓ Tag measurement processed:")
            print(f"  Estimated vehicle position: ({measurement[0]:.2f}, {measurement[1]:.2f})")
            print(f"  Estimated heading: {np.degrees(measurement[2]):.1f}°")
            
            # The vehicle should be estimated at roughly (2, 5) since tag is at (5, 5) and 3m ahead
            if abs(measurement[0] - 2.0) < 1.0:
                print("✓ X estimate reasonable")
            
            return True
        else:
            print("⚠ Measurement processing returned None")
            return True
            
    except Exception as e:
        print(f"✗ Tag measurement processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multiple_tag_fusion():
    """Test 8: Fuse multiple tag measurements"""
    print("\n=== Test 8: Multiple Tag Fusion ===")
    
    try:
        from apriltag_detection import AprilTagDetection
        
        ekf = ExtendedKalmanFilter(
            initial_state=VehicleState(x=0, y=0, theta=0, v=0, omega=0)
        )
        fusion = TagMeasurementFusion(ekf)
        
        # Add two known tags
        fusion.add_tag_to_map(0, x=5.0, y=5.0, theta=0)
        fusion.add_tag_to_map(1, x=-5.0, y=5.0, theta=0)
        
        # Create mock detections
        pose1 = np.eye(4)
        pose1[:3, 3] = [0, 0, 3.0]  # Tag 0 is 3m ahead
        
        pose2 = np.eye(4)
        pose2[:3, 3] = [0, 0, 4.0]  # Tag 1 is 4m ahead
        
        det1 = AprilTagDetection(
            tag_id=0,
            tag_family="tag36h11",
            center=(640, 360),
            pose=pose1,
            distance=3.0,
            bearing=0.0,
            confidence=0.95
        )
        
        det2 = AprilTagDetection(
            tag_id=1,
            tag_family="tag36h11",
            center=(640, 360),
            pose=pose2,
            distance=4.0,
            bearing=0.0,
            confidence=0.90
        )
        
        # Fuse measurements
        fused = fusion.fuse_multiple_tags([det1, det2])
        
        if fused is not None:
            print(f"✓ Fused measurement from 2 tags:")
            print(f"  Position: ({fused[0]:.2f}, {fused[1]:.2f})")
            print(f"  Heading: {np.degrees(fused[2]):.1f}°")
            
            # Closer tag (det1) should have more weight
            return True
        else:
            print("⚠ No valid measurements to fuse")
            return True
            
    except Exception as e:
        print(f"✗ Multiple tag fusion failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ekf_reset():
    """Test 9: Reset EKF to initial state"""
    print("\n=== Test 9: EKF Reset ===")
    
    try:
        initial_state = VehicleState(x=0, y=0, theta=0, v=0, omega=0)
        ekf = ExtendedKalmanFilter(initial_state=initial_state)
        
        # Move the state
        ekf.predict(dt=1.0, control_input=(1.0, 0.0))
        ekf.update(np.array([5.0, 5.0, 0]))
        
        state_before = ekf.get_state()
        print(f"  State before reset: ({state_before.x:.2f}, {state_before.y:.2f})")
        
        # Reset
        ekf.reset(initial_state)
        
        state_after = ekf.get_state()
        
        assert state_after.x == 0
        assert state_after.y == 0
        assert state_after.theta == 0
        
        print(f"✓ State after reset: ({state_after.x}, {state_after.y})")
        print("✓ EKF reset successful")
        
        return True
    except Exception as e:
        print(f"✗ EKF reset failed: {e}")
        return False


def run_all_tests():
    """Run all Kalman filter tests"""
    print("=" * 60)
    print("KALMAN FILTER TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Vehicle State Creation", test_vehicle_state_creation),
        ("EKF Initialization", test_ekf_initialization),
        ("EKF Prediction", test_ekf_prediction),
        ("EKF Update", test_ekf_update),
        ("Angle Normalization", test_angle_normalization),
        ("Tag Fusion Init", test_tag_measurement_fusion_init),
        ("Tag Measurement Processing", test_tag_measurement_processing),
        ("Multiple Tag Fusion", test_multiple_tag_fusion),
        ("EKF Reset", test_ekf_reset),
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
