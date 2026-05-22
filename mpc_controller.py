"""
Lightweight Geometric Path Controller
Pure Pursuit + P-Control + Obstacle Braking
Operates in camera-frame to eliminate EKF drift and support moving targets.
Outputs normalized commands [-1.0, 1.0] for RC car integration.
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class ControllerConfig:
    dt: float = 0.1
    max_velocity: float = 1.5         # m/s
    max_acceleration: float = 0.8     # m/s^2
    max_steer_angle: float = 0.8      # rad (approx 45 deg)
    wheelbase: float = 0.5            # m
    lookahead_dist: float = 0.8       # m
    obstacle_safety_margin: float = 0.3 # m
    obstacle_slowdown_dist: float = 1.0 # m

class PathFollowingController:
    def __init__(self, config: ControllerConfig):
        self.config = config
        self.current_state = np.zeros(5)  # x, y, theta, v, steering_angle
        
    def update_state(self, state_array: np.ndarray):
        """Update controller with latest EKF state estimate."""
        self.current_state = state_array.copy()
        
    def compute_control(self, path: list, obstacles: list) -> Tuple[float, float]:
        """
        Compute normalized acceleration and steering commands.
        Returns: (accel_cmd [-1, 1], steer_cmd [-1, 1])
        """
        # Only velocity is needed from the EKF for speed control
        v = self.current_state[3]
        
        # 1. OBSTACLE BRAKING FACTOR
        min_dist = float('inf')
        for obs in obstacles:
            if obs.distance < min_dist:
                min_dist = obs.distance
                
        speed_scale = 1.0
        if min_dist < self.config.obstacle_safety_margin:
            return -1.0, 0.0  # Hard brake, normalized
        elif min_dist < self.config.obstacle_slowdown_dist:
            # Proportional slowing between safety margin and slowdown distance
            speed_scale = (min_dist - self.config.obstacle_safety_margin) / \
                          (self.config.obstacle_slowdown_dist - self.config.obstacle_safety_margin)
            speed_scale = max(0.0, speed_scale)
            
        # 2. PURE PURSUIT (Camera Frame)
        # Camera frame convention (OpenCV): X=lateral(right), Y=vertical(down), Z=forward
        target_fwd, target_lat = None, None
        for pt in path:
            if hasattr(pt, 'end'):
                lat, _, fwd = pt.end[0], pt.end[1], pt.end[2]
            else:
                lat, fwd = pt[0], pt[1] # Fallback for 2D arrays
            
            dist = np.hypot(fwd, lat)
            if dist >= self.config.lookahead_dist:
                target_fwd, target_lat = fwd, lat
                break
                
        if target_fwd is None:
            # No valid waypoint: brake smoothly to a stop
            brake_cmd = np.clip(-v / (self.config.max_acceleration * self.config.dt), -1.0, 0.0)
            return brake_cmd, 0.0
            
        # 3. HEADING ERROR & STEERING CONTROL
        # In camera frame, car is at (0,0) facing +Z. Target angle is atan2(lateral, forward).
        alpha = np.arctan2(target_lat, target_fwd)
        
        curvature = (2.0 * np.sin(alpha)) / self.config.lookahead_dist
        target_steering = np.clip(curvature * self.config.wheelbase, 
                                  -self.config.max_steer_angle, 
                                  self.config.max_steer_angle)
        
        # 4. SPEED CONTROL
        dist_to_target = np.hypot(target_fwd, target_lat)
        target_v = np.clip(dist_to_target * 1.5, 0.0, self.config.max_velocity) * speed_scale
        
        v_error = target_v - v
        accel = np.clip(1.2 * v_error, -self.config.max_acceleration, self.config.max_acceleration)
        
        # 5. NORMALIZE OUTPUTS TO [-1, 1]
        accel_cmd = accel / self.config.max_acceleration
        steer_cmd = target_steering / self.config.max_steer_angle
        
        return np.clip(accel_cmd, -1.0, 1.0), np.clip(steer_cmd, -1.0, 1.0)