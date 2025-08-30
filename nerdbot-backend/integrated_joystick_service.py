"""
Integrated Joystick Service for NerdBot

This service provides zero-latency joystick control by directly integrating with
motor and servo control modules, eliminating HTTP API overhead.
"""

import logging
import signal
import sys
import time
from joystick_input_manager import get_input_manager


class IntegratedJoystickService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        self.input_manager = get_input_manager()
        self.running = False
        
        # Set up graceful shutdown
        signal.signal(signal.SIGINT, self._shutdown_handler)
        signal.signal(signal.SIGTERM, self._shutdown_handler)
    
    def _shutdown_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.stop()
        sys.exit(0)
    
    def _enable_motor_control(self) -> bool:
        """Enable direct motor control integration"""
        try:
            from motor_control import motors
            success = motors.enable_joystick_control()
            if success:
                self.logger.info("Motor control integration enabled")
            else:
                self.logger.error("Failed to enable motor control integration")
            return success
        except ImportError as e:
            self.logger.error(f"Motor control module not available: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error enabling motor control: {e}")
            return False
    
    def _enable_servo_control(self) -> bool:
        """Enable direct servo control integration"""
        try:
            from servo_control import servos
            success = servos.enable_joystick_control()
            if success:
                self.logger.info("Servo control integration enabled")
            else:
                self.logger.error("Failed to enable servo control integration")
            return success
        except ImportError as e:
            self.logger.error(f"Servo control module not available: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error enabling servo control: {e}")
            return False
    
    def start(self) -> bool:
        """Start the integrated joystick service"""
        self.logger.info("Starting integrated joystick service...")
        
        # Enable control integrations
        motor_ok = self._enable_motor_control()
        servo_ok = self._enable_servo_control()
        
        if not (motor_ok or servo_ok):
            self.logger.error("No control modules could be enabled")
            return False
        
        # Start the input manager
        if not self.input_manager.start():
            self.logger.error("Failed to start input manager")
            return False
        
        self.running = True
        self.logger.info("Integrated joystick service started successfully")
        self.logger.info("Steam Controller mapping:")
        self.logger.info("  Left analog stick: Pan/tilt camera")
        self.logger.info("  Left touchpad: Robot movement (WASD)")
        self.logger.info("  Right touchpad: Robot rotation (radial speed control)")
        self.logger.info("  A button: Center servos")
        self.logger.info("  B button: Emergency stop motors")
        self.logger.info("Press Ctrl+C to stop")
        
        return True
    
    def run(self):
        """Main service loop"""
        if not self.start():
            return False
        
        try:
            # Keep the main thread alive while input manager runs in background
            while self.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.logger.info("Shutdown requested by user")
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the service"""
        if not self.running:
            return
            
        self.logger.info("Stopping integrated joystick service...")
        self.running = False
        
        # Stop the input manager
        self.input_manager.stop()
        
        # Stop motors for safety
        try:
            from motor_control import motors
            motors.stop()
        except ImportError:
            pass
        
        self.logger.info("Integrated joystick service stopped")


def main():
    """Main entry point"""
    service = IntegratedJoystickService()
    service.run()


if __name__ == "__main__":
    main()