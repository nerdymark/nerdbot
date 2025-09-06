#!/usr/bin/env python3
"""
NerdBot Service Manager
Provides easy management of the nerdbot-flask service with GPIO handling
"""
import sys
import subprocess
import time
import json
import requests
from pathlib import Path

class ServiceManager:
    def __init__(self):
        self.service_name = "nerdbot-flask"
        self.api_base = "http://localhost:5000/api"
        
    def get_service_status(self):
        """Get systemd service status"""
        try:
            result = subprocess.run(
                ["sudo", "systemctl", "is-active", self.service_name],
                capture_output=True,
                text=True
            )
            return result.stdout.strip()
        except Exception:
            return "unknown"
    
    def stop_service(self):
        """Stop the service with proper cleanup"""
        print("🛑 Stopping nerdbot-flask service...")
        
        # First try API stop (includes GPIO cleanup)
        if self.get_service_status() == "active":
            try:
                response = requests.post(f"{self.api_base}/service/stop", timeout=10)
                if response.status_code == 200:
                    print("✅ Service stopped via API (with GPIO cleanup)")
                    return True
            except Exception as e:
                print(f"⚠️  API stop failed: {e}, trying systemctl...")
        
        # Fallback to systemctl
        try:
            subprocess.run(
                ["sudo", "systemctl", "stop", self.service_name],
                check=True,
                timeout=10
            )
            print("✅ Service stopped via systemctl")
            return True
        except Exception as e:
            print(f"❌ Failed to stop service: {e}")
            return False
    
    def start_service(self):
        """Start the service"""
        print("🚀 Starting nerdbot-flask service...")
        try:
            subprocess.run(
                ["sudo", "systemctl", "start", self.service_name],
                check=True,
                timeout=15
            )
            
            # Wait a bit and check status
            time.sleep(2)
            status = self.get_service_status()
            if status == "active":
                print("✅ Service started successfully")
                return True
            else:
                print(f"⚠️  Service status: {status}")
                return False
                
        except Exception as e:
            print(f"❌ Failed to start service: {e}")
            return False
    
    def restart_service(self):
        """Restart the service with proper cleanup"""
        print("🔄 Restarting nerdbot-flask service...")
        
        # First try API restart (includes GPIO cleanup)
        if self.get_service_status() == "active":
            try:
                response = requests.post(f"{self.api_base}/service/restart", timeout=15)
                if response.status_code == 200:
                    print("✅ Service restarted via API (with GPIO cleanup)")
                    return True
            except Exception as e:
                print(f"⚠️  API restart failed: {e}, trying manual restart...")
        
        # Manual restart: stop then start
        if self.stop_service():
            time.sleep(1)
            return self.start_service()
        return False
    
    def reset_gpio(self, pin=21):
        """Reset a GPIO pin"""
        print(f"🔧 Resetting GPIO {pin}...")
        
        # Try via API first
        if self.get_service_status() == "active":
            try:
                data = {"pin": pin}
                response = requests.post(
                    f"{self.api_base}/gpio/reset", 
                    json=data, 
                    timeout=5
                )
                if response.status_code == 200:
                    result = response.json()
                    print(f"✅ {result.get('message', 'GPIO reset successful')}")
                    return True
            except Exception as e:
                print(f"⚠️  API GPIO reset failed: {e}, trying direct method...")
        
        # Direct method using our utilities
        try:
            result = subprocess.run(
                ["sudo", "python3", str(Path(__file__).parent / "force_reset_gpio.py")],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                print("✅ GPIO reset completed")
                return True
            else:
                print(f"❌ GPIO reset failed: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ GPIO reset error: {e}")
            return False
    
    def check_gpio_status(self, pin=21):
        """Check GPIO pin status"""
        print(f"📊 Checking GPIO {pin} status...")
        
        # Try via API first
        if self.get_service_status() == "active":
            try:
                response = requests.get(f"{self.api_base}/gpio/status?pin={pin}", timeout=5)
                if response.status_code == 200:
                    result = response.json()
                    print(f"GPIO {pin}: {result.get('status', 'Unknown')}")
                    return True
            except Exception as e:
                print(f"⚠️  API status check failed: {e}, trying direct method...")
        
        # Direct method
        try:
            result = subprocess.run(
                ["gpioinfo", "gpiochip0"],
                capture_output=True,
                text=True,
                timeout=5
            )
            for line in result.stdout.split('\\n'):
                if f"line {pin:>3}:" in line:
                    print(f"GPIO {pin}: {line.strip()}")
                    return True
            print(f"GPIO {pin}: Not found")
            return False
        except Exception as e:
            print(f"❌ GPIO status check error: {e}")
            return False
    
    def full_restart(self):
        """Complete restart with GPIO cleanup"""
        print("🔄 Performing full service restart with GPIO cleanup...")
        
        # Stop service
        if not self.stop_service():
            return False
        
        # Reset GPIO
        self.reset_gpio(21)
        
        # Wait a moment
        time.sleep(1)
        
        # Start service
        return self.start_service()
    
    def status(self):
        """Show comprehensive status"""
        print("📊 NerdBot Service Status")
        print("=" * 40)
        
        # Service status
        status = self.get_service_status()
        print(f"Service: {status}")
        
        # GPIO status
        self.check_gpio_status(21)
        
        # API health check
        if status == "active":
            try:
                response = requests.get(f"{self.api_base}/health", timeout=3)
                if response.status_code == 200:
                    print("API: ✅ Healthy")
                else:
                    print(f"API: ⚠️  HTTP {response.status_code}")
            except Exception:
                print("API: ❌ Not responding")
        else:
            print("API: ⚠️  Service not active")

def main():
    """CLI interface"""
    manager = ServiceManager()
    
    if len(sys.argv) < 2:
        print("Usage: python3 service_manager.py [command]")
        print()
        print("Commands:")
        print("  status    - Show service and GPIO status")
        print("  start     - Start the service") 
        print("  stop      - Stop the service (with cleanup)")
        print("  restart   - Restart the service (with cleanup)")
        print("  reset-gpio [pin] - Reset GPIO pin (default: 21)")
        print("  full-restart - Stop, reset GPIO, start")
        return 1
    
    command = sys.argv[1]
    
    if command == "status":
        manager.status()
    elif command == "start":
        manager.start_service()
    elif command == "stop":
        manager.stop_service()
    elif command == "restart":
        manager.restart_service()
    elif command == "reset-gpio":
        pin = int(sys.argv[2]) if len(sys.argv) > 2 else 21
        manager.reset_gpio(pin)
    elif command == "full-restart":
        manager.full_restart()
    else:
        print(f"Unknown command: {command}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())