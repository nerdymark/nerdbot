"""
Optimized pan/tilt tracking for bounding boxes
"""
import time
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class OptimizedTracker:
    """
    Optimized tracker with proportional control and smooth movement
    """
    
    def __init__(self, servos, frame_width=640, frame_height=480):
        self.servos = servos
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Dead zone parameters (pixels) - increased for more locked-in tracking
        self.dead_zone_x = 60  # Horizontal dead zone (reduced for more responsive tracking)
        self.dead_zone_y = 45  # Vertical dead zone (reduced for more responsive tracking)
        
        # Speed parameters - optimized for smoother movement
        self.max_speed_pan = 1.5    # Max degrees per update (smoother)
        self.max_speed_tilt = 1.0   # Max degrees per update (smoother)
        
        # Proportional gains - optimized for smooth tracking
        self.kp_pan = 0.008   # Horizontal proportional gain (reduced)
        self.kp_tilt = 0.006  # Vertical proportional gain (reduced)
        
        # Center of frame
        self.center_x = frame_width // 2
        self.center_y = frame_height // 2
        
        # Smoothing parameters - optimized for stable tracking
        self.smooth_factor = 0.85  # How much to blend with previous position (increased)
        self.last_error_x = 0
        self.last_error_y = 0
        
    def track_object(self, bbox: Tuple[int, int, int, int], look_at_top: bool = False, object_type: str = None) -> Tuple[float, float]:
        """
        Track an object given its bounding box
        
        Args:
            bbox: Tuple of (x, y, width, height)
            look_at_top: If True, look at top of bounding box (for eye contact)
            object_type: Type of object being tracked (e.g., 'person', 'cat', 'dog')
            
        Returns:
            Tuple of (pan_angle, tilt_angle)
        """
        x, y, w, h = bbox
        
        # Calculate object center
        obj_center_x = x + w // 2
        
        # Smart positioning based on object type
        if look_at_top and object_type == 'person':
            # For people, look at top 1/6 for eye contact
            obj_center_y = y + h // 6
        elif object_type in ['cat', 'dog', 'bird']:
            # For animals, look at center or slightly above center (head area)
            obj_center_y = y + h // 3  # Look at top third (head area)
        elif look_at_top:
            # For other objects where look_at_top is requested, be more conservative
            obj_center_y = y + h // 4  # Look at top quarter instead of top sixth
        else:
            # Default to center
            obj_center_y = y + h // 2
        
        # Calculate error from frame center
        error_x = obj_center_x - self.center_x
        error_y = obj_center_y - self.center_y
        
        # Apply smoothing
        error_x = self.smooth_factor * error_x + (1 - self.smooth_factor) * self.last_error_x
        error_y = self.smooth_factor * error_y + (1 - self.smooth_factor) * self.last_error_y
        
        self.last_error_x = error_x
        self.last_error_y = error_y
        
        # Check if within dead zone
        if abs(error_x) < self.dead_zone_x and abs(error_y) < self.dead_zone_y:
            return self.current_pan, self.current_tilt
        
        # Calculate proportional control output
        pan_delta = self.kp_pan * error_x  # Positive - if object is right of center, pan right
        tilt_delta = self.kp_tilt * error_y  # Positive - normal operation (pulse width handles inversion)
        
        # Apply speed limits
        pan_delta = max(-self.max_speed_pan, min(self.max_speed_pan, pan_delta))
        tilt_delta = max(-self.max_speed_tilt, min(self.max_speed_tilt, tilt_delta))
        
        # Get current actual servo positions
        try:
            current_angles = self.servos.get_current_angles()
            current_pan = current_angles['pan']
            current_tilt = current_angles['tilt']
        except:
            # Fallback to defaults if servo reading fails
            current_pan = 90
            current_tilt = 150
        
        # Calculate new positions
        new_pan = current_pan + pan_delta
        new_tilt = current_tilt + tilt_delta
        
        # Apply servo limits
        new_pan = max(0, min(180, new_pan))
        # Tilt servo safe range: up=90, forward=150, so limit between 80-160
        new_tilt = max(80, min(160, new_tilt))
        
        # Move servos
        try:
            self.servos.pan_to(new_pan)
        except ValueError as e:
            logger.warning(f"Pan servo error: {e}")
            
        try:
            self.servos.tilt_to(new_tilt)
        except ValueError as e:
            logger.warning(f"Tilt servo error: {e}")
            
        return new_pan, new_tilt
    
    def reset(self):
        """Reset servos to center position"""
        self.servos.reset_servos()
        # Reset smoothing history
        self.last_error_x = 0
        self.last_error_y = 0
        
    def set_speed(self, pan_speed: float, tilt_speed: float):
        """Update maximum speeds"""
        self.max_speed_pan = pan_speed
        self.max_speed_tilt = tilt_speed
        
    def set_gains(self, kp_pan: float, kp_tilt: float):
        """Update proportional gains"""
        self.kp_pan = kp_pan
        self.kp_tilt = kp_tilt
        
    def set_dead_zone(self, dead_zone_x: int, dead_zone_y: int):
        """Update dead zone parameters"""
        self.dead_zone_x = dead_zone_x
        self.dead_zone_y = dead_zone_y


class PredictiveTracker(OptimizedTracker):
    """
    Tracker with predictive capabilities for smoother movement
    """
    
    def __init__(self, servos, frame_width=640, frame_height=480):
        super().__init__(servos, frame_width, frame_height)
        
        # History for prediction
        self.position_history = []
        self.history_size = 5
        
        # Prediction parameters
        self.prediction_factor = 0.3
        
    def track_object(self, bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        """
        Track object with prediction
        """
        x, y, w, h = bbox
        
        # Calculate object center
        obj_center_x = x + w // 2
        obj_center_y = y + h // 2
        
        # Add to history
        self.position_history.append((obj_center_x, obj_center_y))
        if len(self.position_history) > self.history_size:
            self.position_history.pop(0)
            
        # Predict future position if we have enough history
        if len(self.position_history) >= 3:
            # Simple linear prediction
            dx = self.position_history[-1][0] - self.position_history[-3][0]
            dy = self.position_history[-1][1] - self.position_history[-3][1]
            
            # Predicted position
            pred_x = obj_center_x + dx * self.prediction_factor
            pred_y = obj_center_y + dy * self.prediction_factor
            
            # Use predicted position for tracking
            obj_center_x = pred_x
            obj_center_y = pred_y
            
        # Calculate error from frame center
        error_x = obj_center_x - self.center_x
        error_y = obj_center_y - self.center_y
        
        # Continue with normal tracking
        return super().track_object((obj_center_x - w//2, obj_center_y - h//2, w, h))