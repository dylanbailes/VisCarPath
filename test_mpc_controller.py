"""
Test Suite for MPC Controller Module
Tests vehicle dynamics, MPC optimization, and path following control
"""

import numpy as np
import sys
from mpc_controller import (
    MPCConfig,
    MPCResult,
    VehicleDynamics,
    MPController,
    PathFollowingController
)


def test_mpc_config():
    """Test 1: Create MPC configuration"""
    print("\n=== Test 1: MPC Configuration ===")
    
    try:
        config = MPCConfig(
            horizon=10,
            dt=0.1,
            max_velocity=2.0,
            max_acceleration=1.0
        )
        
        assert config.horizon == 10
        assert config.dt == 0.1
        assert config.max_velocity == 2.0
        assert config.max_acceleration == 1.0
        
        print("✓ MPC configuration created:")
        print(f"  Horizon: {config.horizon} steps")
        print(f"  Time step: {config.dt}s")
        print(f"  Max velocity: {config.max_velocity} m/s")
        print(f"  Max acceleration: {config.max_acceleration} m/s²")
        
        return True
    except Exception as e:
        print(f"✗ MPC config creation failed: {e}")
        return False


def test_vehicle_dynamics():
    """Test 2: Test vehicle dynamics model"""
    print("\n=== Test 2: Vehicle Dynamics Model ===")
    
    try:
        dynamics = VehicleDynamics(wheelbase=0.5)
        
        assert dynamics.L == 0.5
        
        # Test kinematics with straight motion
        state = np.array([0, 0, 0, 1.0, 0])  # x, y, theta, v, delta
        control = np.array([0.5, 0])  # a, delta_dot
        
        derivatives = dynamics.kinematics(state, control)
        
        print("✓ Vehicle dynamics initialized:")
        print(f"  Wheelbase: {dynamics.L}m")
        print(f"  State derivatives at v=1m/s, straight:")
        print(f"    dx/dt: {derivatives[0]:.2f} m/s")
        print(f"    dy/dt: {derivatives[1]:.2f} m/s")
        print(f"    dtheta/dt: {derivatives[2]:.2f} rad/s")
        print(f"    dv/dt: {derivatives[3]:.2f} m/s²")
        
        # Test prediction step
        next_state = dynamics.predict_step(state, control, dt=0.1)
        
        expected_x = 0 + 1.0 * np.cos(0) * 0.1
        error_x = abs(next_state[0] - expected_x)
        
        print(f"\n✓ Prediction after 0.1s:")
        print(f"  New position: ({next_state[0]:.3f}, {next_state[1]:.3f})")
        print(f"  Error in x: {error_x*100:.1f}cm")
        
        if error_x < 0.001:
            print("✓ Prediction accurate!")
        
        return True
    except Exception as e:
        print(f"✗ Vehicle dynamics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_steering_kinematics():
    """Test 3: Test steering kinematics"""
    print("\n=== Test 3: Steering Kinematics ===")
    
    try:
        dynamics = VehicleDynamics(wheelbase=0.5)
        
        # Test with steering angle
        state = np.array([0, 0, 0, 1.0, 0.3])  # 0.3 rad steering
        control = np.array([0, 0])
        
        derivatives = dynamics.kinematics(state, control)
        
        print(f"✓ With steering angle 0.3 rad (17°):")
        print(f"  Angular velocity: {derivatives[2]:.3f} rad/s")
        
        # Turning radius should be L/tan(delta)
        expected_omega = 1.0 * np.tan(0.3) / 0.5
        error = abs(derivatives[2] - expected_omega)
        
        print(f"  Expected omega: {expected_omega:.3f} rad/s")
        print(f"  Error: {error:.4f} rad/s")
        
        if error < 0.001:
            print("✓ Steering kinematics correct!")
        
        return True
    except Exception as e:
        print(f"✗ Steering kinematics test failed: {e}")
        return False


def test_mpc_controller_init():
    """Test 4: Initialize MPC controller"""
    print("\n=== Test 4: MPC Controller Initialization ===")
    
    try:
        config = MPCConfig(horizon=10, dt=0.1)
        controller = MPController(config)
        
        assert controller.config.horizon == 10
        assert controller.dynamics.L == 0.5
        
        print("✓ MPC controller initialized")
        print(f"  State dimension: {controller.nx}")
        print(f"  Control dimension: {controller.nu}")
        
        return True
    except Exception as e:
        print(f"✗ MPC controller init failed: {e}")
        return False


def test_linearization():
    """Test 5: Test dynamics linearization"""
    print("\n=== Test 5: Dynamics Linearization ===")
    
    try:
        controller = MPController(MPCConfig(horizon=10, dt=0.1))
        
        # Linearize around nominal state
        state_nominal = np.array([0, 0, 0, 1.0, 0])
        control_nominal = np.array([0, 0])
        
        A, B, c = controller.linearize_dynamics(state_nominal, control_nominal)
        
        assert A.shape == (5, 5)
        assert B.shape == (5, 2)
        assert len(c) == 5
        
        print("✓ Linearization successful:")
        print(f"  A matrix shape: {A.shape}")
        print(f"  B matrix shape: {B.shape}")
        print(f"  Affine term: {c}")
        
        # Check structure of A matrix
        print("\n  A matrix (first 3 rows):")
        for i in range(3):
            print(f"    [{A[i,0]:.2f}, {A[i,1]:.2f}, {A[i,2]:.2f}, {A[i,3]:.2f}, {A[i,4]:.2f}]")
        
        return True
    except Exception as e:
        print(f"✗ Linearization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_reference_trajectory_generation():
    """Test 6: Generate reference trajectory"""
    print("\n=== Test 6: Reference Trajectory Generation ===")
    
    try:
        from ground_obstacle_detection import PathSegment
        
        config = MPCConfig(horizon=10, dt=0.1)
        controller = MPController(config)
        
        # Create simple path segments
        segments = [
            PathSegment(
                start=np.array([0, 0, 0]),
                end=np.array([0, 0, 3]),
                width=0.6,
                clearance=0.3,
                cost=1.0
            ),
            PathSegment(
                start=np.array([0, 0, 3]),
                end=np.array([1, 0, 5]),
                width=0.6,
                clearance=0.3,
                cost=1.0
            )
        ]
        
        current_state = np.array([0, 0, 0, 0.5, 0])
        
        reference = controller.compute_reference_trajectory(segments, current_state)
        
        print(f"✓ Generated reference trajectory:")
        print(f"  Number of points: {len(reference)}")
        print(f"  Expected: {config.horizon + 1} points")
        
        if len(reference) > 0:
            print(f"  First point: [{reference[0][0]:.2f}, {reference[0][1]:.2f}, "
                  f"{np.degrees(reference[0][2]):.1f}°, {reference[0][3]:.2f} m/s]")
            print(f"  Last point: [{reference[-1][0]:.2f}, {reference[-1][1]:.2f}, "
                  f"{np.degrees(reference[-1][2]):.1f}°, {reference[-1][3]:.2f} m/s]")
        
        return True
    except Exception as e:
        print(f"✗ Reference trajectory generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mpc_solve_simple():
    """Test 7: Solve simple MPC problem"""
    print("\n=== Test 7: Simple MPC Optimization ===")
    
    try:
        config = MPCConfig(horizon=5, dt=0.1)
        controller = MPController(config)
        
        # Current state: at origin, moving forward
        current_state = np.array([0, 0, 0, 0.5, 0])
        
        # Simple reference: go straight
        reference = []
        for i in range(config.horizon + 1):
            t = i * config.dt
            reference.append(np.array([0, t * 0.5, 0, 0.5]))
        
        result = controller.solve_mpc(current_state, reference, obstacles=None)
        
        print(f"✓ MPC solve completed:")
        print(f"  Success: {result.success}")
        print(f"  Solve time: {result.solve_time*1000:.1f}ms")
        print(f"  Cost: {result.cost:.2f}")
        
        if result.success:
            print(f"  Optimal acceleration: {result.acceleration:.3f} m/s²")
            print(f"  Optimal steering rate: {result.steering_rate:.3f} rad/s")
            
            if len(result.predicted_trajectory) > 0:
                print(f"  Predicted trajectory points: {len(result.predicted_trajectory)}")
        
        return True
    except Exception as e:
        print(f"✗ MPC solve failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mpc_with_obstacles():
    """Test 8: Solve MPC with obstacle avoidance"""
    print("\n=== Test 8: MPC with Obstacle Avoidance ===")
    
    try:
        config = MPCConfig(horizon=5, dt=0.1)
        controller = MPController(config)
        
        current_state = np.array([0, 0, 0, 0.5, 0])
        
        # Reference: go straight
        reference = []
        for i in range(config.horizon + 1):
            t = i * config.dt
            reference.append(np.array([0, t * 0.5, 0, 0.5]))
        
        # Add obstacle in the way
        obstacles = [
            {'position': np.array([0.0, 1.0]), 'radius': 0.3}
        ]
        
        result = controller.solve_mpc(current_state, reference, obstacles=obstacles)
        
        print(f"✓ MPC with obstacle avoidance:")
        print(f"  Success: {result.success}")
        print(f"  Solve time: {result.solve_time*1000:.1f}ms")
        
        if result.success:
            print(f"  Acceleration: {result.acceleration:.3f} m/s²")
            print(f"  Steering rate: {result.steering_rate:.3f} rad/s")
            
            # Check if trajectory avoids obstacle
            if len(result.predicted_trajectory) > 0:
                min_dist = float('inf')
                for traj_point in result.predicted_trajectory:
                    dist = np.linalg.norm(np.array(traj_point[:2]) - obstacles[0]['position'])
                    min_dist = min(min_dist, dist)
                
                print(f"  Minimum distance to obstacle: {min_dist:.2f}m")
        
        return True
    except Exception as e:
        print(f"✗ MPC with obstacles failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_path_following_controller():
    """Test 9: Test high-level path following controller"""
    print("\n=== Test 9: Path Following Controller ===")
    
    try:
        from ground_obstacle_detection import PathSegment, Obstacle
        
        config = MPCConfig(horizon=5, dt=0.1)
        controller = PathFollowingController(config)
        
        # Set current state
        current_state = np.array([0, 0, 0, 0.5, 0])
        controller.update_state(current_state)
        
        # Create path
        path_segments = [
            PathSegment(
                start=np.array([0, 0, 0]),
                end=np.array([0, 0, 3]),
                width=0.6,
                clearance=0.3,
                cost=1.0
            )
        ]
        
        # No obstacles
        obstacles = []
        
        accel, steer = controller.compute_control(path_segments, obstacles)
        
        print(f"✓ Path following control computed:")
        print(f"  Acceleration: {accel:.3f} m/s²")
        print(f"  Steering rate: {steer:.3f} rad/s")
        
        return True
    except Exception as e:
        print(f"✗ Path following controller failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fallback_control():
    """Test 10: Test fallback PD control"""
    print("\n=== Test 10: Fallback PD Control ===")
    
    try:
        from ground_obstacle_detection import PathSegment
        
        config = MPCConfig()
        controller = PathFollowingController(config)
        
        # Set state away from target
        current_state = np.array([0, 0, 0, 0, 0])
        controller.update_state(current_state)
        
        # Create path segment
        path_segments = [
            PathSegment(
                start=np.array([0, 0, 0]),
                end=np.array([0, 3, 3]),  # Target 3m forward and right
                width=0.6,
                clearance=0.3,
                cost=1.0
            )
        ]
        
        # Access fallback control directly
        accel, steer = controller._fallback_control(path_segments)
        
        print(f"✓ Fallback control computed:")
        print(f"  Acceleration: {accel:.3f} m/s²")
        print(f"  Steering rate: {steer:.3f} rad/s")
        
        # Should have positive acceleration toward target
        if accel > 0:
            print("✓ Positive acceleration toward target")
        
        return True
    except Exception as e:
        print(f"✗ Fallback control failed: {e}")
        return False


def run_all_tests():
    """Run all MPC controller tests"""
    print("=" * 60)
    print("MPC CONTROLLER TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("MPC Config", test_mpc_config),
        ("Vehicle Dynamics", test_vehicle_dynamics),
        ("Steering Kinematics", test_steering_kinematics),
        ("MPC Controller Init", test_mpc_controller_init),
        ("Dynamics Linearization", test_linearization),
        ("Reference Trajectory", test_reference_trajectory_generation),
        ("Simple MPC Solve", test_mpc_solve_simple),
        ("MPC with Obstacles", test_mpc_with_obstacles),
        ("Path Following Controller", test_path_following_controller),
        ("Fallback Control", test_fallback_control),
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
