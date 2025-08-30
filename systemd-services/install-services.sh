#!/bin/bash
# Install NerdBot systemd services
# Run as: sudo bash install-services.sh

set -e

if [ "$EUID" -ne 0 ]; then
    echo "Please run as root: sudo bash install-services.sh"
    exit 1
fi

echo "Installing NerdBot systemd services..."

# Copy service files to systemd directory
cp /home/mark/systemd-services/*.service /etc/systemd/system/

# Set proper ownership and permissions
chown root:root /etc/systemd/system/nerdbot-*.service
chmod 644 /etc/systemd/system/nerdbot-*.service

# Reload systemd
systemctl daemon-reload

echo ""
echo "Services installed successfully!"
echo ""
echo "To enable and start all services:"
echo "  sudo systemctl enable nerdbot-fan nerdbot-flask nerdbot-ui nerdbot-tts nerdbot-joystick nerdbot-darkice"
echo "  sudo systemctl start nerdbot-fan nerdbot-flask nerdbot-ui nerdbot-tts nerdbot-joystick nerdbot-darkice"
echo ""
echo "To check service status:"
echo "  sudo systemctl status nerdbot-flask"
echo "  journalctl -u nerdbot-flask -f"
echo ""
echo "To stop all services:"
echo "  sudo systemctl stop nerdbot-fan nerdbot-flask nerdbot-ui nerdbot-tts nerdbot-joystick nerdbot-darkice"
echo ""
echo "To disable services from auto-start:"
echo "  sudo systemctl disable nerdbot-fan nerdbot-flask nerdbot-ui nerdbot-tts nerdbot-joystick nerdbot-darkice"